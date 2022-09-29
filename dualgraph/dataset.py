from torch_geometric.data import InMemoryDataset
from ogb.lsc import PygPCQM4MDataset
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data
import torch
import os.path as osp
from torch_sparse import SparseTensor
import re
import os
import shutil
from ogb.utils.url import decide_download, download_url, extract_zip
from dualgraph.mol import smiles2graphwithface
import numpy as np
import io


class DGPygPCQM4MDataset(PygPCQM4MDataset):
    def __init__(self, root, smiles2graph=None, transform=None, pre_transform=None):
        super().__init__(
            root=root, smiles2graph=smiles2graph, transform=transform, pre_transform=pre_transform
        )

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, "data.csv.gz"))
        smiles_list = data_df["smiles"]
        homolumogap_list = data_df["homolumogap"]

        print("Converting SMILES strings into graphs...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = DGData()

            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]
            graph = self.smiles2graph(smiles)

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.y = torch.Tensor([homolumogap])

            data.ring_mask = torch.from_numpy(graph["ring_mask"]).to(torch.bool)
            data.ring_index = torch.from_numpy(graph["ring_index"]).to(torch.int64)
            data.nf_node = torch.from_numpy(graph["nf_node"]).to(torch.int64)
            data.nf_ring = torch.from_numpy(graph["nf_ring"]).to(torch.int64)
            data.num_rings = int(graph["num_rings"])
            data.n_edges = int(graph["n_edges"])
            data.n_nodes = int(graph["n_nodes"])
            data.n_nfs = int(graph["n_nfs"])
            data_list.append(data)

        # double-check prediction target
        split_dict = self.get_idx_split()
        assert all([not torch.isnan(data_list[i].y)[0] for i in split_dict["train"]])
        assert all([not torch.isnan(data_list[i].y)[0] for i in split_dict["valid"]])
        assert all([torch.isnan(data_list[i].y)[0] for i in split_dict["test"]])

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def download(self):
        return super().download()


class DGData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if isinstance(value, SparseTensor):
            return (0, 1)
        elif bool(re.search("(index|face)", key)):
            return -1
        elif bool(re.search("(nf_node|nf_ring)", key)):
            return -1
        return 0

    def __inc__(self, key, value, *args, **kwargs):
        if bool(re.search("(ring_index|nf_ring)", key)):
            return int(self.num_rings.item())
        elif bool(re.search("(index|face|nf_node)", key)):
            return self.num_nodes
        else:
            return 0


class DGPygGraphPropPredDataset(InMemoryDataset):
    def __init__(self, name, root="dataset", transform=None, pre_transform=None):
        self.name = name
        self.dir_name = "_".join(name.split("-")) + "_ring"
        self.original_root = root
        self.root = osp.join(root, self.dir_name)

        master = pd.read_csv(os.path.join(os.path.dirname(__file__), "master.csv"), index_col=0)
        if not self.name in master:
            error_mssg = "Invalid dataset name {}.\n".format(self.name)
            error_mssg += "Available datasets are as follows:\n"
            error_mssg += "\n".join(master.keys())
            raise ValueError(error_mssg)
        self.meta_info = master[self.name]

        if osp.isdir(self.root) and (
            not osp.exists(
                osp.join(self.root, "RELEASE_v" + str(self.meta_info["version"]) + ".txt")
            )
        ):
            print(self.name + " has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.root)

        self.download_name = self.meta_info["download_name"]

        self.num_tasks = int(self.meta_info["num tasks"])
        self.eval_metric = self.meta_info["eval metric"]
        self.task_type = self.meta_info["task type"]
        self.__num_classes__ = int(self.meta_info["num classes"])
        self.binary = self.meta_info["binary"] == "True"

        super().__init__(self.root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_idx_split(self, split_type=None):
        if split_type is None:
            split_type = self.meta_info["split"]

        path = osp.join(self.root, "split", split_type)

        if os.path.isfile(os.path.join(path, "split_dict.pt")):
            return torch.load(os.path.join(path, "split_dict.pt"))

        train_idx = pd.read_csv(
            osp.join(path, "train.csv.gz"), compression="gzip", header=None
        ).values.T[0]
        valid_idx = pd.read_csv(
            osp.join(path, "valid.csv.gz"), compression="gzip", header=None
        ).values.T[0]
        test_idx = pd.read_csv(
            osp.join(path, "test.csv.gz"), compression="gzip", header=None
        ).values.T[0]
        return {
            "train": torch.tensor(train_idx, dtype=torch.long),
            "valid": torch.tensor(valid_idx, dtype=torch.long),
            "test": torch.tensor(test_idx, dtype=torch.long),
        }

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        if self.binary:
            return ["data.npz"]
        else:
            file_names = ["edge"]
            if self.meta_info["has_node_attr"] == "True":
                file_names.append("node-feat")
            if self.meta_info["has_edge_attr"] == "True":
                file_names.append("edge-feat")
            return [file_name + ".csv.gz" for file_name in file_names]

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        url = self.meta_info["url"]
        if decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root, self.download_name), self.root)

        else:
            print("Stop downloading.")
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        smiles_list = pd.read_csv(osp.join(self.root, "mapping", "mol.csv.gz"), compression="gzip")[
            "smiles"
        ].values
        labels = pd.read_csv(
            osp.join(self.raw_dir, "graph-label.csv.gz"), compression="gzip", header=None
        ).values
        has_nan = np.isnan(labels).any()

        print("Converting SMILES strings into graphs...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = DGData()
            smiles = smiles_list[i]
            graph = smiles2graphwithface(smiles)

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            if "classification" in self.task_type and (not has_nan):
                data.y = torch.from_numpy(labels[i]).view(1, -1).to(torch.long)
            else:
                data.y = torch.from_numpy(labels[i]).view(1, -1).to(torch.float32)

            data.ring_mask = torch.from_numpy(graph["ring_mask"]).to(torch.bool)
            data.ring_index = torch.from_numpy(graph["ring_index"]).to(torch.int64)
            data.nf_node = torch.from_numpy(graph["nf_node"]).to(torch.int64)
            data.nf_ring = torch.from_numpy(graph["nf_ring"]).to(torch.int64)
            data.num_rings = int(graph["num_rings"])
            data.n_edges = int(graph["n_edges"])
            data.n_nodes = int(graph["n_nodes"])
            data.n_nfs = int(graph["n_nfs"])
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])


class BinaryRawDataset(InMemoryDataset):
    def __init__(self, name, root="dataset", transform=None, pre_transform=None, base_path=None):
        self.name = name
        self.dirname = "_".join(name.split("-"))
        self.original_root = root
        self.root = osp.join(root, self.dirname)
        self.base_path = base_path
        super().__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_classes(self):
        return 2

    @property
    def raw_file_names(self):
        return "data.csv.gz"

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self,):
        assert os.path.exists(self.base_path)

    def process(self):
        bad_case = 0
        train_idx = []
        valid_idx = []
        test_idx = []
        data_list = []
        for subset in ["train", "valid", "test"]:

            try:
                with io.open(f"{subset}.smi", "r", encoding="utf8", newline="\n") as f:
                    lines = f.readlines()
                smiles_list = [line.strip() for line in lines]
                with io.open(f"{subset}.label", "r", encoding="utf8", newline="\n") as f:
                    lines = f.readlines()
                labels = [int(x.strip()) for x in lines]
                assert len(smiles_list) == len(labels)
                for i in tqdm(range(len(smiles_list))):
                    data = DGData()
                    smiles = smiles_list[i]
                    graph = smiles2graphwithface(smiles)

                    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                    assert len(graph["node_feat"]) == graph["num_nodes"]

                    data.__num_nodes__ = int(graph["num_nodes"])
                    data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)

                    data.y = torch.from_numpy(labels[i]).view(1, -1).to(torch.long)
                    data.ring_mask = torch.from_numpy(graph["ring_mask"]).to(torch.bool)
                    data.ring_index = torch.from_numpy(graph["ring_index"]).to(torch.int64)
                    data.nf_node = torch.from_numpy(graph["nf_node"]).to(torch.int64)
                    data.nf_ring = torch.from_numpy(graph["nf_ring"]).to(torch.int64)
                    data.num_rings = int(graph["num_rings"])
                    data.n_edges = int(graph["n_edges"])
                    data.n_nodes = int(graph["n_nodes"])
                    data.n_nfs = int(graph["n_nfs"])

                    if subset == "train":
                        train_idx.append(len(data_list))
                    elif subset == "valid":
                        valid_idx.append(len(data_list))
                    else:
                        test_idx.append(len(data_list))

                    data_list.append(data)

            except:
                bad_case += 1

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

        print(f"valid {len(data_list)} bad case {bad_case}")

