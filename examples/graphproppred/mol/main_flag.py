from numpy.core.records import array
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR
from dualgraph.mol import smiles2graphwithface
from dualgraph.gnn import GNN2, GNNwithvn
import os
from tqdm import tqdm
import argparse
import time
import numpy as np
import random
import io
from dualgraph.utils import WarmCosine, WarmLinear
import json

from dualgraph.dataset import DGPygGraphPropPredDataset
from ogb.graphproppred import Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()


def train_vanilla(model, device, loader, optimizer, task_type, scheduler, args):
    model.train()
    loss_accum = 0

    pbar = tqdm(loader, desc="Iteration")
    for step, batch in enumerate(pbar):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type:
                loss = cls_criterion(
                    pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled]
                )
            else:
                loss = reg_criterion(
                    pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled]
                )
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_accum += loss.detach().item()
            if step % args.log_interval == 0:
                pbar.set_description(
                    "Iteration loss: {:6.4f} lr: {:.5e}".format(
                        loss_accum / (step + 1), scheduler.get_last_lr()[0]
                    )
                )


def train(model, device, loader, optimizer, task_type, scheduler, args):
    loss_accum = 0
    pbar = tqdm(loader, desc="Iteration")
    for step, batch in enumerate(pbar):
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            is_labeled = batch.y == batch.y

            forward = lambda perturb, perturb_edge: model(batch, perturb, perturb_edge).to(
                torch.float32
            )[is_labeled]
            model_forward = (model, forward)
            y = batch.y.to(torch.float32)[is_labeled]
            perturb_shape = (batch.x.shape[0], args.latent_size)
            perturb_edge_shape = (batch.edge_attr.shape[0], args.latent_size)
            loss = flag(
                model_forward,
                perturb_shape,
                perturb_edge_shape,
                y,
                args,
                optimizer,
                device,
                cls_criterion,
                scheduler,
            )

            loss_accum += loss.detach().item()
            if step % args.log_interval == 0:
                pbar.set_description(
                    "Iteration loss: {:6.4f} lr: {:.5e}".format(
                        loss_accum / (step + 1), scheduler.get_last_lr()[0]
                    )
                )


def flag(
    model_forward,
    perturb_shape,
    perturb_edge_shape,
    y,
    args,
    optimizer,
    device,
    criterion,
    scheduler,
):

    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    perturb = torch.FloatTensor(*perturb_shape).uniform_(-args.step_size, args.step_size).to(device)
    perturb.requires_grad_()
    if args.attack_edge:
        perturb_edge = (
            torch.FloatTensor(*perturb_edge_shape)
            .uniform_(-args.step_size, args.step_size)
            .to(device)
        )
        perturb_edge.requires_grad_()
    else:
        perturb_edge = None
    out = forward(perturb, perturb_edge)
    loss = criterion(out, y)
    loss /= args.m

    for _ in range(args.m - 1):
        loss.backward()
        perturb_data = perturb.detach() + args.step_size * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0

        if args.attack_edge:
            perturb_edge_data = perturb_edge.detach() + args.step_size * torch.sign(
                perturb_edge.grad.detach()
            )
            perturb_edge.data = perturb_edge_data.data
            perturb_edge.grad[:] = 0

        out = forward(perturb, perturb_edge)
        loss = criterion(out, y)
        loss /= args.m

    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss * args.m


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():
    parser = argparse.ArgumentParser(
        description="GNN baselines on ogbgmol* data with Pytorch Geometrics"
    )
    parser.add_argument("--device", type=int, default=0, help="which gpu to use ")
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-dir", type=str, default="", help="tensorboard log directory")
    parser.add_argument("--checkpoint-dir", type=str, default="")
    parser.add_argument("--save-test-dir", type=str, default="")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dropnet", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--period", type=float, default=10)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--use-outer", action="store_true", default=False)
    parser.add_argument("--lr-warmup", action="store_true", default=False)
    parser.add_argument("--residual", action="store_true", default=False)
    parser.add_argument("--layernorm-before", action="store_true", default=False)
    parser.add_argument("--parallel", action="store_true", default=False)

    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    parser.add_argument("--from-pretrained", type=str, default="")
    parser.add_argument("--layer-drop", type=float, default=0.0)
    parser.add_argument("--pooler-dropout", type=float, default=0.0)
    parser.add_argument("--use-vn", action="store_true", default=False)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--encoder-dropout", type=float, default=0.0)
    parser.add_argument("--use-bn", action="store_true", default=False)
    parser.add_argument("--use-adamw", action="store_true", default=False)
    parser.add_argument("--node-attn", action="store_true", default=False)
    parser.add_argument("--face-attn", action="store_true", default=False)
    parser.add_argument("--global-attn", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step-size", type=float, default=8e-3)
    parser.add_argument("-m", type=int, default=3)
    parser.add_argument("--attack-edge", action="store_true", default=False)

    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    dataset = DGPygGraphPropPredDataset(name=args.dataset)
    split_idx = dataset.get_idx_split()

    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(
        dataset[split_idx["train"]],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        dataset[split_idx["valid"]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        dataset[split_idx["test"]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

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
        "num_tasks": dataset.num_tasks,
        "layer_drop": args.layer_drop,
        "pooler_dropout": args.pooler_dropout,
        "encoder_dropout": args.encoder_dropout,
        "use_bn": args.use_bn,
        "node_attn": args.node_attn,
        "face_attn": args.face_attn,
        "global_attn": args.global_attn,
        "flag": True,
    }
    if args.use_vn:
        model = GNNwithvn(**shared_params).to(device)
    else:
        model = GNN2(**shared_params).to(device)
    if args.from_pretrained:
        assert os.path.exists(args.from_pretrained)
        checkpoint = torch.load(args.from_pretrained, map_location=torch.device("cpu"))[
            "model_state_dict"
        ]
        keys_to_delete = []
        for k, v in checkpoint.items():
            if "decoder" in k:
                keys_to_delete.append(k)
        for k in keys_to_delete:
            print(f"delete {k} from pre-trained checkpoint...")
            del checkpoint[k]

        for k, v in model.state_dict().items():
            if k not in checkpoint:
                print(f"randomly init {k}...")
                checkpoint[k] = v
        model.load_state_dict(checkpoint)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"#Params: {num_params}")
    if args.use_adamw:
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, betas=(0.9, args.beta2), weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, betas=(0.9, args.beta2), weight_decay=args.weight_decay
        )
    # if args.from_pretrained:
    #     lrscheduler = WarmLinear(
    #         tmax=len(train_loader) * args.period, warmup=len(train_loader) * args.period * 0.06
    #     )
    #     scheduler = LambdaLR(optimizer, lambda x: lrscheduler.step(x))
    # elif not args.lr_warmup:
    if not args.lr_warmup:
        scheduler = LambdaLR(optimizer, lambda x: 1.0)
    else:
        lrscheduler = WarmCosine(tmax=len(train_loader) * args.period, warmup=int(4e3))
        scheduler = LambdaLR(optimizer, lambda x: lrscheduler.step(x))

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print("Training...")
        train(model, device, train_loader, optimizer, dataset.task_type, scheduler, args)

        print("Evaluating...")
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)
        if args.checkpoint_dir:
            print(f"settings {os.path.basename(args.checkpoint_dir)}...")
        print({"Train": train_perf, "Validation": valid_perf, "Test": test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

        if args.checkpoint_dir:
            logs = {
                "Train": train_perf[dataset.eval_metric],
                "Validation": valid_perf[dataset.eval_metric],
                "Test": test_perf[dataset.eval_metric],
            }
            with io.open(
                os.path.join(args.checkpoint_dir, "log.txt"), "a", encoding="utf8", newline="\n"
            ) as tgt:
                print(json.dumps(logs), file=tgt)

    if "classification" in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print("Finished training!")
    print("Best validation score: {}".format(valid_curve[best_val_epoch]))
    print("Test score: {}".format(test_curve[best_val_epoch]))


if __name__ == "__main__":
    main()
