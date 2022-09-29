import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR
from dualgraph.mol import smiles2graphwithface
from dualgraph.gnn import GNN, GNN2
import os
from tqdm import tqdm
import argparse
import time
import numpy as np
import random
import io
from dualgraph.utils import WarmCosine, init_distributed_mode
import json


### importing OGB-LSC
from ogb.lsc import PCQM4MEvaluator
from dualgraph.dataset import DGPygPCQM4MDataset
from torch.utils.data import DistributedSampler

reg_criterion = torch.nn.L1Loss()


def train(model, device, loader, optimizer, scheduler, args):
    model.train()
    loss_accum = 0

    pbar = tqdm(loader, desc="Iteration", disable=args.disable_tqdm)
    for step, batch in enumerate(pbar):
        batch = batch.to(device)

        pred = model(batch).view(-1,)
        optimizer.zero_grad()
        loss = reg_criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_accum += loss.detach().cpu().item()
        if step % args.log_interval == 0:
            pbar.set_description(
                "Iteration loss: {:6.4f} lr: {:.5e}".format(
                    loss_accum / (step + 1), scheduler.get_last_lr()[0]
                )
            )

    return loss_accum / (step + 1)


def eval(model, device, loader, evaluator, args):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration", disable=args.disable_tqdm)):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]


def test(model, device, loader):
    model.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)

    return y_pred


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="GNN baselines on pcqm4m with Pytorch Geometrics")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gnn", type=str, default="dualgraph2")
    parser.add_argument("--global-reducer", type=str, default="sum")
    parser.add_argument("--node-reducer", type=str, default="sum")
    parser.add_argument("--face-reducer", type=str, default="sum")
    parser.add_argument("--graph-pooling", type=str, default="sum")
    parser.add_argument("--init-face", action="store_true", default=False)
    parser.add_argument("--dropedge-rate", type=float, default=0.1)
    parser.add_argument("--dropnode-rate", type=float, default=0.1)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--latent-size", type=int, default=256)
    parser.add_argument("--mlp-hidden-size", type=int, default=1024)
    parser.add_argument("--mlp_layers", type=int, default=2)
    parser.add_argument("--use-layer-norm", action="store_true", default=False)
    parser.add_argument("--ignore-face", action="store_true", default=False)
    parser.add_argument("--use-global", action="store_true", default=False)
    parser.add_argument("--train-subset", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-dir", type=str, default="", help="tensorboard log directory")
    parser.add_argument("--checkpoint-dir", type=str, default="")
    parser.add_argument("--save-test-dir", type=str, default="")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dropnet", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--period", type=float, default=10)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--use-outer", action="store_true", default=False)
    parser.add_argument("--lr-warmup", action="store_true", default=False)
    parser.add_argument("--residual", action="store_true", default=False)
    parser.add_argument("--layernorm-before", action="store_true", default=False)
    parser.add_argument("--parallel", action="store_true", default=False)
    parser.add_argument("--use-bn", action="store_true", default=False)
    parser.add_argument("--node-attn", action="store_true", default=False)
    parser.add_argument("--face-attn", action="store_true", default=False)
    parser.add_argument("--global-attn", action="store_true", default=False)
    parser.add_argument("--use-adamw", action="store_true", default=False)
    parser.add_argument("--pooler-dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--reload", action="store_true", default=False)

    args = parser.parse_args()
    init_distributed_mode(args)
    print(args)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    device = torch.device(args.device)

    ### automatic dataloading and splitting
    dataset = DGPygPCQM4MDataset(root="dataset/", smiles2graph=smiles2graphwithface)

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = PCQM4MEvaluator()

    if args.train_subset:
        subset_ratio = 0.1
        subset_idx = torch.randperm(len(split_idx["train"]))[
            : int(subset_ratio * len(split_idx["train"]))
        ]
        train_loader = DataLoader(
            dataset[split_idx["train"][subset_idx]],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
    else:
        dataset_train = dataset[split_idx["train"]]
        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True
        )

        train_loader = DataLoader(
            dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers,
        )
        # train_loader = DataLoader(
        #     dataset[split_idx["train"]],
        #     batch_size=args.batch_size,
        #     shuffle=True,
        #     num_workers=args.num_workers,
        # )

    valid_loader = DataLoader(
        dataset[split_idx["valid"]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    if args.save_test_dir is not "":
        test_loader = DataLoader(
            dataset[split_idx["test"]],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    if args.checkpoint_dir is not "":
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    shared_params = {
        "mlp_hidden_size": args.mlp_hidden_size,
        "mlp_layers": args.mlp_layers,
        "latent_size": args.latent_size,
        "use_layer_norm": args.use_layer_norm,
        "num_message_passing_steps": args.num_layers,
        "global_reducer": args.global_reducer,
        "node_reducer": args.node_reducer,
        "face_reducer": args.face_reducer,
        "dropedge_rate": args.dropedge_rate,
        "dropnode_rate": args.dropnode_rate,
        "ignore_globals": not args.use_global,
        "use_face": not args.ignore_face,
        "dropout": args.dropout,
        "dropnet": args.dropnet,
        "init_face": args.init_face,
        "graph_pooling": args.graph_pooling,
        "use_outer": args.use_outer,
        "residual": args.residual,
        "layernorm_before": args.layernorm_before,
        "parallel": args.parallel,
        "pooler_dropout": args.pooler_dropout,
        "use_bn": args.use_bn,
        "node_attn": args.node_attn,
        "face_attn": args.face_attn,
        "global_attn": args.global_attn,
    }

    if args.gnn == "dualgraph":
        model = GNN(**shared_params).to(device)
    elif args.gnn == "dualgraph2":
        model = GNN2(**shared_params).to(device)
    else:
        raise ValueError("Invalid GNN type")

    model_without_ddp = model
    args.disable_tqdm = False
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        model_without_ddp = model.module

        args.checkpoint_dir = "" if args.rank != 0 else args.checkpoint_dir
        # args.enable_tb = False if args.rank != 0 else args.enable_tb
        args.disable_tqdm = args.rank != 0

    num_params = sum(p.numel() for p in model_without_ddp.parameters())
    print(f"#Params: {num_params}")

    if args.use_adamw:
        optimizer = optim.AdamW(
            model_without_ddp.parameters(),
            lr=args.lr,
            betas=(0.9, args.beta2),
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.Adam(
            model_without_ddp.parameters(),
            lr=args.lr,
            betas=(0.9, args.beta2),
            weight_decay=args.weight_decay,
        )

    if args.log_dir is not "":
        writer = SummaryWriter(log_dir=args.log_dir)

    best_valid_mae = 1000

    if args.train_subset:
        scheduler = StepLR(optimizer, step_size=300, gamma=0.25)
        args.epochs = 1000
    else:
        if not args.lr_warmup:
            scheduler = CosineAnnealingLR(optimizer, len(train_loader) * args.period, 1e-7)
        else:
            lrscheduler = WarmCosine(tmax=len(train_loader) * args.period)
            scheduler = LambdaLR(optimizer, lambda x: lrscheduler.step(x))
    _start_epoch = 0
    if args.reload:
        print("Reload from {}...".format(os.path.join(args.checkpoint_dir, "checkpoint.pt")))
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, "checkpoint.pt"))
        _start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_valid_mae = checkpoint["best_val_mae"]

    for epoch in range(_start_epoch + 1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print("Training...")
        train_mae = train(model, device, train_loader, optimizer, scheduler, args)

        print("Evaluating...")
        valid_mae = eval(model, device, valid_loader, evaluator, args)

        print({"Train": train_mae, "Validation": valid_mae})

        if args.log_dir is not "":
            writer.add_scalar("valid/mae", valid_mae, epoch)
            writer.add_scalar("train/mae", train_mae, epoch)

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            if args.checkpoint_dir is not "":
                print("Saving checkpoint to {}...".format(args.checkpoint_dir))
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model_without_ddp.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_mae": best_valid_mae,
                    "num_params": num_params,
                }
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, "checkpoint.pt"))

            if args.save_test_dir is not "":
                print("Predicting on test data...")
                y_pred = test(model, device, test_loader)
                print("Saving test submission file...")
                evaluator.save_test_submission({"y_pred": y_pred}, args.save_test_dir)

        # scheduler.step()
        logs = {
            "epoch": epoch,
            "train_mae": train_mae,
            "valie_mae": valid_mae,
            "best_valid_mae": best_valid_mae,
        }
        with io.open(
            os.path.join(args.checkpoint_dir, "log.txt"), "a", encoding="utf8", newline="\n"
        ) as tgt:
            print(json.dumps(logs), file=tgt)
        print(f"Best validation MAE so far: {best_valid_mae}")

    if args.log_dir is not "":
        writer.close()
    if args.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
