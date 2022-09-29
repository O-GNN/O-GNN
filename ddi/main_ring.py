import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from dualgraph.gnn import GNN2
import os
from tqdm import tqdm
import argparse
import numpy as np
import random
import io
from dualgraph.utils import WarmCosine, WarmLinear
import json

from ddi.dataset import DDIDataset
from ddi.evaluate import Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()


def train(model, device, loader, optimizer, scheduler, args):
    model.train()
    loss_accum = 0

    pbar = tqdm(loader, desc="Iteration")
    for step, batch in enumerate(pbar):
        batch_a, batch_b, labels = batch
        batch_a = batch_a.to(device)
        batch_b = batch_b.to(device)
        labels = labels.to(device)

        hidden_a = model(batch_a)
        hidden_b = model(batch_b)
        hidden = torch.cat([hidden_a, hidden_b], dim=-1)
        pred = model.get_last_layer(hidden)

        optimizer.zero_grad()
        is_labeled = labels == labels
        loss = cls_criterion(
            pred.to(torch.float32)[is_labeled], labels.to(torch.float32)[is_labeled]
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

    return loss_accum / (step + 1)


def evaluate(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc="Iteration"):
        batch_a, batch_b, labels = batch
        batch_a = batch_a.to(device)
        batch_b = batch_b.to(device)
        with torch.no_grad():
            hidden_a = model(batch_a)
            hidden_b = model(batch_b)
            hidden = torch.cat([hidden_a, hidden_b], dim=-1)
            pred = model.get_last_layer(hidden)

        y_true.append(labels)
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.evaluate(input_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="which gpu to use ")
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-dir", type=str, default="")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--period", type=float, default=10)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lr-warmup", action="store_true", default=False)
    parser.add_argument("--layernorm-before", action="store_true", default=False)

    parser.add_argument("--dataset", type=str, default="inductive_newb3")
    parser.add_argument("--data-path", type=str, default="ddi/data/inductive/new_build3")
    parser.add_argument("--pooler-dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--encoder-dropout", type=float, default=0.0)
    parser.add_argument("--use-bn", action="store_true", default=False)
    parser.add_argument("--use-adamw", action="store_true", default=False)
    parser.add_argument("--node-attn", action="store_true", default=False)
    parser.add_argument("--face-attn", action="store_true", default=False)
    parser.add_argument("--global-attn", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-tb", action="store_true", default=False)

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

    train_dataset = DDIDataset(name=args.dataset, path=args.data_path, split="train")
    valid_dataset = DDIDataset(name=args.dataset, path=args.data_path, split="valid")

    evaluator = Evaluator()
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
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
        "ignore_globals": True,
        "use_face": not args.ignore_face,
        "dropout": args.dropout,
        "dropnet": 0.0,
        "init_face": args.init_face,
        "graph_pooling": args.graph_pooling,
        "use_outer": False,
        "residual": False,
        "layernorm_before": args.layernorm_before,
        "parallel": False,
        "num_tasks": train_dataset.num_tasks,
        "layer_drop": 0.0,
        "pooler_dropout": args.pooler_dropout,
        "encoder_dropout": args.encoder_dropout,
        "use_bn": args.use_bn,
        "node_attn": args.node_attn,
        "face_attn": args.face_attn,
        "global_attn": args.global_attn,
        "ddi": True,
    }
    model = GNN2(**shared_params).to(device)

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

    if not args.lr_warmup:
        lrscheduler = WarmLinear(
            tmax=len(train_loader) * args.epochs, warmup=len(train_loader) * args.epochs * 0.06
        )
    else:
        warmup_step = int(4e3)
        lrscheduler = WarmCosine(tmax=len(train_loader) * args.period, warmup=warmup_step)
    scheduler = LambdaLR(optimizer, lambda x: lrscheduler.step(x))

    if args.checkpoint_dir and args.enable_tb:
        tb_writer = SummaryWriter(args.checkpoint_dir)

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print("Training...")
        loss = train(model, device, train_loader, optimizer, scheduler, args)
        print("Evaluating...")
        train_pref = evaluate(model, device, train_loader, evaluator)
        valid_pref = evaluate(model, device, valid_loader, evaluator)

        if args.checkpoint_dir:
            print(f"Setting {os.path.basename(os.path.normpath(args.checkpoint_dir))}...")
        print(json.dumps(dict(Train=train_pref, Valid=valid_pref)))
        if args.checkpoint_dir:
            logs = dict(epoch=epoch, Train=train_pref, Valid=valid_pref)
            with io.open(
                os.path.join(args.checkpoint_dir, "log.txt"), "a", encoding="utf8", newline="\n"
            ) as tgt:
                print(json.dumps(logs), file=tgt)
            if args.enable_tb:
                tb_writer.add_scalar("train/loss", loss, epoch)
                for k, v in train_pref.items():
                    tb_writer.add_scalar(f"evaluation/train_{k}", v, epoch)
                for k, v in valid_pref.items():
                    tb_writer.add_scalar(f"evaluation/valid_{k}", v, epoch)

    if args.checkpoint_dir and args.enable_tb:
        tb_writer.close()
    print("Finished traning!")


if __name__ == "__main__":
    main()
