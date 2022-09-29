import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from dualgraph.mol import smiles2graphwithface
from dualgraph.gnn import GIN_node_Virtual

import os
from tqdm import tqdm
import argparse
import time
import numpy as np
import random


### importing OGB-LSC
from ogb.lsc import PCQM4MEvaluator
from dualgraph.dataset import DGPygPCQM4MDataset

reg_criterion = torch.nn.L1Loss()


def train(model, device, loader, optimizer, args):
    model.train()
    loss_accum = 0

    pbar = tqdm(loader, desc="Iteration")
    for step, batch in enumerate(pbar):
        batch = batch.to(device)

        pred = model(batch).view(-1,)
        optimizer.zero_grad()
        loss = reg_criterion(pred, batch.y)
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()
        if step % args.log_interval == 0:
            pbar.set_description("Iteration loss: {:6.4f}".format(loss_accum / (step + 1)))

    return loss_accum / (step + 1)


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
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
    parser.add_argument("--device", type=int, default=0, help="which gpu to use ")
    parser.add_argument("--gnn", type=str, default="dualgraph")
    parser.add_argument("--global-reducer", type=str, default="mean")
    parser.add_argument("--init-face", action="store_true", default=False)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--latent-size", type=int, default=600)
    parser.add_argument("--ignore-face", action="store_true", default=False)
    parser.add_argument("--train-subset", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-dir", type=str, default="", help="tensorboard log directory")
    parser.add_argument("--checkpoint-dir", type=str, default="")
    parser.add_argument("--save-test-dir", type=str, default="")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dropnet", type=float, default=0.1)
    args = parser.parse_args()

    print(args)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

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
        "emb_dim": args.latent_size,
        "num_layers": args.num_layers,
        "graph_pooling": args.global_reducer,
        "use_face": not args.ignore_face,
        "dropout": args.dropout,
        "dropnet": args.dropnet,
        "init_face": args.init_face,
        "residual": True,
    }

    if args.gnn == "dualgraph":
        model = GIN_node_Virtual(**shared_params).to(device)
    else:
        raise ValueError("Invalid GNN type")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"#Params: {num_params}")

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.log_dir is not "":
        writer = SummaryWriter(log_dir=args.log_dir)

    best_valid_mae = 1000

    if args.train_subset:
        scheduler = StepLR(optimizer, step_size=300, gamma=0.25)
        args.epochs = 1000
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.25)

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print("Training...")
        train_mae = train(model, device, train_loader, optimizer, args)

        print("Evaluating...")
        valid_mae = eval(model, device, valid_loader, evaluator)

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
                    "model_state_dict": model.state_dict(),
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

        scheduler.step()

        print(f"Best validation MAE so far: {best_valid_mae}")

    if args.log_dir is not "":
        writer.close()


if __name__ == "__main__":
    main()
