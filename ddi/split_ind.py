import os
import pandas as pd
import re
from rdkit import Chem
from tqdm import tqdm
import numpy as np


target_dir = os.path.join(os.path.dirname(__file__), "data", "inductive")
ssiddi_dir = os.path.join(os.path.dirname(__file__), "data", "ssiddi")

# transductive file
trans_smiles = "{}/{}".format(ssiddi_dir, "drug_smiles.csv")
trans_train = "{}/{}".format(ssiddi_dir, "ddi_training.csv")
trans_valid = "{}/{}".format(ssiddi_dir, "ddi_validation.csv")
trans_test = "{}/{}".format(ssiddi_dir, "ddi_test.csv")

# inductive file
ind_smiles = "{}/{}".format(ssiddi_dir, "drugbank_id_smiles.csv")
ind_train = "{}/{}".format(ssiddi_dir, "pair_pos_neg_triples-fold2-train.csv")
ind_valid = "{}/{}".format(ssiddi_dir, "pair_pos_neg_triples-fold2-s2.csv")
ind_test = "{}/{}".format(ssiddi_dir, "pair_pos_neg_triples-fold2-s1.csv")


def check_similes_dict():
    # if all drug in smiles file ?
    trans_id2smiles = {}
    trans_smiles_csv = pd.read_csv(trans_smiles)
    for did, smiles in zip(tqdm(trans_smiles_csv["drug_id"]), trans_smiles_csv["smiles"]):
        trans_id2smiles[did] = smiles

    for file in [trans_train, trans_valid, trans_test]:
        file_csv = pd.read_csv(file)
        for d1, d2 in zip(tqdm(file_csv["d1"]), file_csv["d2"]):
            if d1 not in trans_id2smiles:
                print("error idx = {} is not in trans dict".format(d1))
            if d2 not in trans_id2smiles:
                print("error idx = {} is not in trans dict".format(d1))

    ind_id2smiles = {}
    ind_smiles_csv = pd.read_csv(ind_smiles)
    for did, smiles in zip(tqdm(ind_smiles_csv["DrugID"]), ind_smiles_csv["Smiles"]):
        ind_id2smiles[did] = smiles

    for file in [ind_train, ind_valid, ind_test]:
        file_csv = pd.read_csv(file)
        for d1, d2 in zip(tqdm(file_csv["Drug1_ID"]), file_csv["Drug2_ID"]):
            if d1 not in ind_id2smiles:
                print("error idx = {} is not in ind dict".format(d1))
            if d2 not in ind_id2smiles:
                print("error idx = {} is not in ind dict".format(d1))
    diff_set = set(trans_id2smiles.keys()) - set(ind_id2smiles.keys())
    print("drug interaction len = {}: {}".format(len(diff_set), diff_set))


def check_ilegal_negsample():

    # neg sample is reasonalbe ?
    drug_pair = set()
    drug_set = set()
    ind_train_csv = pd.read_csv(ind_train)
    for d1, d2 in zip(tqdm(ind_train_csv["Drug1_ID"]), ind_train_csv["Drug2_ID"]):
        drug_pair.add((d1, d2))
        drug_pair.add((d2, d1))
        drug_set.add(d1)
        drug_set.add(d2)

    print("drug dict size = {}".format(len(drug_set)))
    print("drug pair size = {}".format(len(drug_pair)))

    replace_error = 0
    keep_error = 0
    for d1, d2, n in zip(
        tqdm(ind_train_csv["Drug1_ID"]), ind_train_csv["Drug2_ID"], ind_train_csv["Neg samples"],
    ):
        d_neg, pos = n.split("$")
        if pos == "h":
            if (d1, d_neg) in drug_pair:
                keep_error += 1
            if (d_neg, d2) in drug_pair:
                replace_error += 1
        else:
            if (d1, d_neg) in drug_pair:
                replace_error += 1
            if (d_neg, d2) in drug_pair:
                keep_error += 1

    print("training set replace error = {}, and keep error = {}".format(replace_error, keep_error))

    ind_valid_csv = pd.read_csv(ind_valid)
    for d1, d2 in zip(tqdm(ind_valid_csv["Drug1_ID"]), ind_valid_csv["Drug2_ID"]):
        drug_pair.add((d1, d2))
        drug_pair.add((d2, d1))

    replace_error = 0
    keep_error = 0
    neg_pair_seen = 0
    drug_unseen = set()
    ind_valid_csv = pd.read_csv(ind_valid)
    for d1, d2, n in zip(
        tqdm(ind_valid_csv["Drug1_ID"]), ind_valid_csv["Drug2_ID"], ind_valid_csv["Neg samples"],
    ):
        d_neg, pos = n.split("$")
        if d1 in drug_set and d2 in drug_set:
            print("known drug occur in valid set")
        if d1 not in drug_set:
            drug_unseen.add(d1)
        else:
            drug_unseen.add(d2)

        if pos == "h":
            assert d2 not in drug_set or d_neg not in drug_set
        else:
            assert d1 not in drug_set or d_neg not in drug_set

        if pos == "h" and d_neg in drug_set and d2 in drug_set:
            neg_pair_seen += 1
        if pos == "t" and d_neg in drug_set and d1 in drug_set:
            neg_pair_seen += 1

        if pos == "h":
            if (d1, d_neg) in drug_pair:
                keep_error += 1
            if (d_neg, d2) in drug_pair:
                replace_error += 1
        else:
            if (d1, d_neg) in drug_pair:
                replace_error += 1
            if (d_neg, d2) in drug_pair:
                keep_error += 1

    print("valid set replace error = {}, and keep error = {}".format(replace_error, keep_error))
    print("valid set neg_pair_seen = {}".format(neg_pair_seen))
    print("valid set drug unseen = {}".format(len(drug_unseen)))


def check_new_build():

    reb_dir = "{}/{}".format(target_dir, "new_build")
    new_ind_train = "{}/{}".format(reb_dir, "train.csv")
    new_ind_valid = "{}/{}".format(reb_dir, "valid.csv")

    drug_pair = set()
    drug_set = set()
    ind_train_csv = pd.read_csv(new_ind_train)
    for d1, d2 in zip(tqdm(ind_train_csv["id1"]), ind_train_csv["id2"]):
        drug_pair.add((d1, d2))
        drug_pair.add((d2, d1))
        drug_set.add(d1)
        drug_set.add(d2)

    print("drug dict size = {}".format(len(drug_set)))
    print("drug pair size = {}".format(len(drug_pair)))

    for nd1, nd2 in zip(tqdm(ind_train_csv["neg_id1"]), ind_train_csv["neg_id2"]):
        assert (nd1, nd2) not in drug_pair
        assert nd1 in drug_set
        assert nd2 in drug_set

    ind_valid_csv = pd.read_csv(new_ind_valid)
    for d1, d2 in zip(tqdm(ind_valid_csv["id1"]), ind_valid_csv["id2"]):
        assert d1 not in drug_set or d2 not in drug_set
        drug_pair.add((d1, d2))
        drug_pair.add((d2, d1))

    for nd1, nd2 in zip(tqdm(ind_valid_csv["neg_id1"]), ind_valid_csv["neg_id2"]):
        assert nd1 not in drug_set or nd2 not in drug_set
        assert (nd1, nd2) not in drug_pair


def check_ssi_ddi():
    # pair number
    # transductive setting 115186 + 38349 + 38338 = 191873
    # inductive setting 123785 + 60656 + 7370 = 191811

    # check_similes_dict()

    # check_ilegal_negsample()

    check_new_build()


def rebuild_drug_pair():

    all_drug_pair = set()
    drug_dict = {}
    train_drug_dict = set()
    for file in [ind_train, ind_valid, ind_test]:
        file_csv = pd.read_csv(file)
        for d1, d2 in zip(tqdm(file_csv["Drug1_ID"]), file_csv["Drug2_ID"]):
            all_drug_pair.add((d1, d2))

            if d1 not in drug_dict:
                drug_dict[d1] = set([d2])
            else:
                drug_dict[d1].add(d2)
            if d2 not in drug_dict:
                drug_dict[d2] = set([d1])
            else:
                drug_dict[d2].add(d1)

            if file == ind_train:
                train_drug_dict.add(d1)
                train_drug_dict.add(d2)

    print("all drug pair len = {}".format(len(all_drug_pair)))
    print("drug_dict len = {}".format(len(drug_dict)))
    print("train_drug_dict len = {}".format(len(train_drug_dict)))

    # limit the negative sample from training set
    all_drug_set = train_drug_dict
    reb_dir = "{}/{}".format(target_dir, "new_build2")
    if not os.path.exists(reb_dir):
        os.makedirs(reb_dir, exist_ok=True)

    new_ind_train = "{}/{}".format(reb_dir, "train.csv")
    new_ind_valid = "{}/{}".format(reb_dir, "valid.csv")
    # new_ind_test = "{}/{}".format(reb_dir, 'test.csv')

    train_data = []
    train_csv = pd.read_csv(ind_train)
    for d1, d2, y in zip(tqdm(train_csv["Drug1_ID"]), train_csv["Drug2_ID"], train_csv["Y"]):
        assert d1 in train_drug_dict and d2 in train_drug_dict

        if np.random.rand() > 0.5:
            neg_d1 = d1
            candidate_list = list(all_drug_set - drug_dict[d1])
            candidate_idx = np.random.choice(len(candidate_list), 1)[-1]
            neg_d2 = candidate_list[candidate_idx]
        else:
            candidate_list = list(all_drug_set - drug_dict[d2])
            candidate_idx = np.random.choice(len(candidate_list), 1)[-1]
            neg_d1 = candidate_list[candidate_idx]
            neg_d2 = d2

        data_dict = {
            "id1": d1,
            "id2": d2,
            "neg_id1": neg_d1,
            "neg_id2": neg_d2,
            "y": y,
        }
        train_data.append(data_dict)

    df_data = pd.DataFrame(train_data)
    df_data.to_csv(new_ind_train)

    valid_data = []
    valid_csv = pd.read_csv(ind_valid)
    for d1, d2, y in zip(tqdm(valid_csv["Drug1_ID"]), valid_csv["Drug2_ID"], valid_csv["Y"]):
        assert d1 not in train_drug_dict or d2 not in train_drug_dict
        if d1 not in train_drug_dict:
            neg_d1 = d1
            candidate_list = list(all_drug_set - drug_dict[d1])
            candidate_idx = np.random.choice(len(candidate_list), 1)[-1]
            neg_d2 = candidate_list[candidate_idx]
            assert neg_d2 in all_drug_set
        else:
            assert d2 not in train_drug_dict
            neg_d2 = d2
            candidate_list = list(all_drug_set - drug_dict[d2])
            candidate_idx = np.random.choice(len(candidate_list), 1)[-1]
            neg_d1 = candidate_list[candidate_idx]
            assert neg_d1 in all_drug_set

        data_dict = {
            "id1": d1,
            "id2": d2,
            "neg_id1": neg_d1,
            "neg_id2": neg_d2,
            "y": y,
        }
        valid_data.append(data_dict)

    df_data = pd.DataFrame(valid_data)
    df_data.to_csv(new_ind_valid)


def main():

    # check_ssi_ddi()

    rebuild_drug_pair()

    # check_ssi_ddi()


if __name__ == "__main__":

    main()
