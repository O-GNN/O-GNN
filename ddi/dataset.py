import os
import os.path as osp
import torch
from torch_geometric.data import Dataset
from tqdm import tqdm
import io
from dualgraph.dataset import DGData
from dualgraph.mol import smiles2graphwithface
import numpy as np


class DDIDataset(Dataset):
    def __init__(
        self,
        name="inductive_newb3",
        root="dataset",
        transform=None,
        pre_transform=None,
        path="ddi/data/inductive/new_build3",
        split="train",
    ):
        self.name = name
        self.split = split
        assert self.split in ["train", "valid"]
        self.dirname = f"ddi_{self.name}_{self.split}"
        self.original_root = root
        self.root = osp.join(root, self.dirname)
        self.base_path = path

        super().__init__(self.root, transform=transform, pre_transform=pre_transform)
        self.data_list_a, self.data_list_b, self.data_label_list, self._num_tasks = torch.load(
            self.processed_paths[0]
        )

    @property
    def raw_dir(self):
        return self.base_path

    @property
    def raw_file_names(self):
        return ["data.npz"]

    @property
    def num_tasks(self):
        return self._num_tasks

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        if not os.path.exists(self.processed_paths[0]):
            assert os.path.exists(os.path.join(self.raw_dir, "train.a"))

    def process(self):
        data_list_a = []
        data_list_b = []
        data_label_list = []

        with io.open(
            os.path.join(self.base_path, "label.dict"), "r", encoding="utf8", newline="\n"
        ) as f:
            label_length = len(f.readlines())

        with io.open(
            os.path.join(self.base_path, f"{self.split}.a"), "r", encoding="utf8", newline="\n"
        ) as f:
            drugs_a_list = f.readlines()

        for drug in tqdm(drugs_a_list):
            drug = drug.strip()
            data_list_a.append(getdata(drug))

        with io.open(
            os.path.join(self.base_path, f"{self.split}.b"), "r", encoding="utf8", newline="\n"
        ) as f:
            drugs_b_list = f.readlines()

        for drug in tqdm(drugs_b_list):
            drug = drug.strip()
            data_list_b.append(getdata(drug))

        with io.open(
            os.path.join(self.base_path, f"{self.split}.label"), "r", encoding="utf8", newline="\n"
        ) as f:
            label_list = f.readlines()

        for label in label_list:
            x = [np.nan] * label_length
            label = int(label)
            x[label] = 1
            data_label_list.append(torch.as_tensor(x).view(-1).to(torch.float32))

        with io.open(
            os.path.join(self.base_path, f"{self.split}.nega"), "r", encoding="utf8", newline="\n"
        ) as f:
            drugs_a_list = f.readlines()

        for drug in tqdm(drugs_a_list):
            drug = drug.strip()
            data_list_a.append(getdata(drug))

        with io.open(
            os.path.join(self.base_path, f"{self.split}.negb"), "r", encoding="utf8", newline="\n"
        ) as f:
            drugs_b_list = f.readlines()

        for drug in tqdm(drugs_b_list):
            drug = drug.strip()
            data_list_b.append(getdata(drug))

        with io.open(
            os.path.join(self.base_path, f"{self.split}.label"), "r", encoding="utf8", newline="\n"
        ) as f:
            label_list = f.readlines()

        for label in label_list:
            x = [np.nan] * label_length
            label = int(label)
            x[label] = 0
            data_label_list.append(torch.as_tensor(x).view(-1).to(torch.float32))

        print("Saving...")
        torch.save(
            (data_list_a, data_list_b, data_label_list, label_length), self.processed_paths[0]
        )

    def len(self):
        return len(self.data_label_list)

    def get(self, idx):
        return (self.data_list_a[idx], self.data_list_b[idx], self.data_label_list[idx])


def getdata(smiles):
    data = DGData()
    graph = smiles2graphwithface(smiles)

    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
    assert len(graph["node_feat"]) == graph["num_nodes"]

    data.__num_nodes__ = int(graph["num_nodes"])
    data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)

    data.ring_mask = torch.from_numpy(graph["ring_mask"]).to(torch.bool)
    data.ring_index = torch.from_numpy(graph["ring_index"]).to(torch.int64)
    data.nf_node = torch.from_numpy(graph["nf_node"]).to(torch.int64)
    data.nf_ring = torch.from_numpy(graph["nf_ring"]).to(torch.int64)
    data.num_rings = torch.tensor(int(graph["num_rings"]))
    data.n_edges = torch.tensor(int(graph["n_edges"]))
    data.n_nodes = torch.tensor(int(graph["n_nodes"]))
    data.n_nfs = torch.tensor(int(graph["n_nfs"]))

    return data

