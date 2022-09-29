import re
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
import os
import pickle as pkl
import numpy as np


def smi_tokenizer(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    try:
        assert re.sub("\s+", "", smi) == "".join(tokens)
    except:
        return ""
    return " ".join(tokens)


def clean_smiles(smiles):
    t = re.sub(":\d*", "", smiles)
    return t


def main():
    print("processing start !")

    curdir = os.path.dirname(__file__)
    raw_data_dir = f"{curdir}/data/ssiddi/"

    inductive_data_dir = f"{curdir}/data/unseen/new_build3/"
    raw_train_dir = "{}{}".format(raw_data_dir, "drugbank_id_smiles.csv")

    all_drug_dict = {}
    drugbank_csv = pd.read_csv(raw_train_dir)
    for idx, smiles in zip(tqdm(drugbank_csv["DrugID"]), drugbank_csv["Smiles"]):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(clean_smiles(smiles)))
        # all_drug_dict[idx] = smi_tokenizer(smiles)
        all_drug_dict[idx] = smiles

    fold0_s2 = "{}{}".format(inductive_data_dir, "valid.csv")
    fold0_train = "{}{}".format(inductive_data_dir, "train.csv")

    train_a_dir = "{}{}".format(inductive_data_dir, "train.a")
    train_b_dir = "{}{}".format(inductive_data_dir, "train.b")
    valid_a_dir = "{}{}".format(inductive_data_dir, "valid.a")
    valid_b_dir = "{}{}".format(inductive_data_dir, "valid.b")

    train_nega_dir = "{}{}".format(inductive_data_dir, "train.nega")
    train_negb_dir = "{}{}".format(inductive_data_dir, "train.negb")
    valid_nega_dir = "{}{}".format(inductive_data_dir, "valid.nega")
    valid_negb_dir = "{}{}".format(inductive_data_dir, "valid.negb")

    train_label_b_dir = "{}{}".format(inductive_data_dir, "train.label")
    valid_label_b_dir = "{}{}".format(inductive_data_dir, "valid.label")

    label_dict = {}
    with open(train_a_dir, "w") as ta_w, open(train_nega_dir, "w") as tna_w, open(
        train_b_dir, "w"
    ) as tb_w, open(train_negb_dir, "w") as tnb_w, open(valid_a_dir, "w") as va_w, open(
        valid_nega_dir, "w"
    ) as vna_w, open(
        valid_b_dir, "w"
    ) as vb_w, open(
        valid_negb_dir, "w"
    ) as vnb_w, open(
        train_label_b_dir, "w"
    ) as tl, open(
        valid_label_b_dir, "w"
    ) as vl:

        train_csv = pd.read_csv(fold0_train)
        for a, b, na, nb, y in zip(
            tqdm(train_csv["id1"]),
            train_csv["id2"],
            train_csv["neg_id1"],
            train_csv["neg_id2"],
            train_csv["y"],
        ):

            ta_w.writelines(all_drug_dict[a] + "\n")
            tb_w.writelines(all_drug_dict[b] + "\n")

            tna_w.writelines(all_drug_dict[na] + "\n")
            tnb_w.writelines(all_drug_dict[nb] + "\n")

            tl.writelines(str(y) + "\n")
            if str(y) not in label_dict:
                label_dict[str(y)] = len(label_dict)

        valid_csv = pd.read_csv(fold0_s2)
        for a, b, na, nb, y in zip(
            tqdm(valid_csv["id1"]),
            valid_csv["id2"],
            valid_csv["neg_id1"],
            valid_csv["neg_id2"],
            valid_csv["y"],
        ):

            va_w.writelines(all_drug_dict[a] + "\n")
            vb_w.writelines(all_drug_dict[b] + "\n")

            vna_w.writelines(all_drug_dict[na] + "\n")
            vnb_w.writelines(all_drug_dict[nb] + "\n")

            vl.writelines(str(y) + "\n")
            if str(y) not in label_dict:
                label_dict[str(y)] = len(label_dict)

    label_dict_dir = "{}{}".format(inductive_data_dir, "label.dict")
    with open(label_dict_dir, "w") as label_dict_w:
        for label_name, label_idx in label_dict.items():
            label_dict_w.writelines(label_name + " " + str(label_idx) + "\n")

    print("processing done !")


if __name__ == "__main__":
    main()
