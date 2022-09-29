from ogb.utils.features import (
    allowable_features,
    atom_to_feature_vector,
    bond_to_feature_vector,
    atom_feature_vector_to_dict,
    bond_feature_vector_to_dict,
)
from rdkit import Chem
import numpy as np
from dualgraph.graph import get2DConformer, Graph, getface


def smiles2graphwithface(smiles_string):

    if not isinstance(smiles_string, Chem.Mol):
        mol = Chem.MolFromSmiles(smiles_string)
    else:
        mol = smiles_string

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = np.array(edges_list, dtype=np.int64).T
        edge_attr = np.array(edge_features_list, dtype=np.int64)

        faces, left, _ = getface(mol)
        num_faces = len(faces)
        face_mask = [False] * num_faces
        face_index = [[-1, -1]] * len(edges_list)
        face_mask[0] = True
        for i in range(len(edges_list)):
            inface = left[i ^ 1]
            outface = left[i]
            face_index[i] = [inface, outface]

        nf_node = []
        nf_ring = []
        for i, face in enumerate(faces):
            face = list(set(face))
            nf_node.extend(face)
            nf_ring.extend([i] * len(face))

        face_mask = np.array(face_mask, dtype=np.bool)
        face_index = np.array(face_index, dtype=np.int64).T
        n_nfs = len(nf_node)
        nf_node = np.array(nf_node, dtype=np.int64).reshape(1, -1)
        nf_ring = np.array(nf_ring, dtype=np.int64).reshape(1, -1)

    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)
        face_mask = np.empty((0), dtype=np.bool)
        face_index = np.empty((2, 0), dtype=np.int64)
        num_faces = 0
        n_nfs = 0
        nf_node = np.empty((1, 0), dtype=np.int64)
        nf_ring = np.empty((1, 0), dtype=np.int64)

    graph = dict()
    graph["edge_index"] = edge_index
    graph["edge_feat"] = edge_attr
    graph["node_feat"] = x
    graph["num_nodes"] = len(x)

    # we do not use the keyword "face", since "face" is already used by torch_geometric.
    graph["ring_mask"] = face_mask
    graph["ring_index"] = face_index
    graph["num_rings"] = num_faces
    graph["n_edges"] = len(edge_attr)
    graph["n_nodes"] = len(x)

    graph["n_nfs"] = n_nfs
    graph["nf_node"] = nf_node
    graph["nf_ring"] = nf_ring

    return graph


if __name__ == "__main__":
    graph = smiles2graphwithface(r"[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CCC2=O")
    print(graph)

