from rdkit import Chem
from rdkit.Chem import rdDepictor
from collections import defaultdict
from math import atan2


def get2DConformer(mol):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)
    conformer = mol.GetConformer()
    return conformer


class Edge:
    def __init__(self, fro, to, ang):
        self._fro = fro
        self._to = to
        self._ang = ang

    @property
    def ang(self):
        return self._ang

    @property
    def fro(self):
        return self._fro

    @property
    def to(self):
        return self._to


class Node:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y


class Graph:
    def __init__(self) -> None:
        self.nodes = dict()
        self.edges = []
        self.neighbors = defaultdict(list)

    def addEdge(self, fro, to):
        self.edges.append(Edge(fro, to, self.get_angle(fro, to)))
        self.neighbors[fro].append(len(self.edges) - 1)
        self.edges.append(Edge(to, fro, self.get_angle(to, fro)))
        self.neighbors[to].append(len(self.edges) - 1)

    def addNode(self, nodeidx, coord):
        self.nodes[nodeidx] = Node(*coord)

    def get_angle(self, fro, to):
        return atan2(self.nodes[to].y - self.nodes[fro].y, self.nodes[to].x - self.nodes[fro].x)

    def findFaces(self):
        prev = [None] * len(self.edges)
        left = [None] * len(self.edges)
        vis = [False] * len(self.edges)

        # sort the edges for each vertex
        for node in self.nodes.keys():
            neighbor_edges = self.neighbors[node]
            edge_num = len(neighbor_edges)
            for i in range(edge_num):
                for j in range(i + 1, edge_num):
                    if self.edges[neighbor_edges[i]].ang > self.edges[neighbor_edges[j]].ang:
                        tmp = neighbor_edges[j]
                        neighbor_edges[j] = neighbor_edges[i]
                        neighbor_edges[i] = tmp

            for i in range(edge_num):
                prev[neighbor_edges[(i + 1) % edge_num]] = neighbor_edges[i]

        face_cnt = -1
        faces = []

        for node in self.nodes.keys():
            for edge in self.neighbors[node]:
                if not vis[edge]:
                    face_cnt += 1
                    face = []
                    source = edge
                    while True:
                        vis[edge] = True
                        left[edge] = face_cnt
                        fro = self.edges[edge].fro
                        face.append(fro)
                        edge = prev[edge ^ 1]
                        if edge == source:
                            break
                        assert vis[edge] == False
                    faces.append(face)
        return faces, left

    def is_intersection(self):
        is_intersection = False
        for i in range(len(self.edges), step=2):
            for j in range(i + 2, len(self.edges), step=2):
                e1 = self.edges[i]
                e2 = self.edges[j]
                node = set([e1.fro, e2.fro, e1.to, e2.to])
                if len(node) < 4:
                    continue
                points = [self.nodes[i] for i in node]


def getConformer(mol):
    assert isinstance(mol, Chem.Mol)
    conformer = get2DConformer(mol)
    G = Graph()

    for atom in mol.GetAtoms():
        atomidx = atom.GetIdx()
        atompos = conformer.GetAtomPosition(atomidx)
        G.addNode(atomidx, [atompos.x, atompos.y])

    if len(mol.GetBonds()) > 0:
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            G.addEdge(i, j)

    faces, left = G.findFaces()
    return faces, left


def getface(mol):
    assert isinstance(mol, Chem.Mol)
    bond2id = dict()
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond2id[(i, j)] = len(bond2id)
        bond2id[(j, i)] = len(bond2id)

    num_edge = len(bond2id)
    left = [0] * num_edge
    ssr = Chem.GetSymmSSSR(mol)
    face = [[]]
    for ring in ssr:
        ring = list(ring)
        
        bond_list = []
        for i, atom in enumerate(ring):
            bond_list.append((ring[i-1], atom))
        
        exist = False
        if any([left[bond2id[bond]] != 0 for bond in bond_list]):
            exist = True 
        if exist:
            ring = list(reversed(ring))
        face.append(ring)
        for i, atom in enumerate(ring):
            bond = (ring[i - 1], atom)
            if left[bond2id[bond]] != 0:
                bond = (atom, ring[i - 1])
            bondid = bond2id[bond]
            if left[bondid] == 0:
                left[bondid] = len(face) - 1

    return face, left, bond2id


if __name__ == "__main__":
    smiles_string = r"[N+]12CCC(CC1)C(OC(=O)C(O)(c1ccccc1)c1ccccc1)C2.[Br-]"
    mol = Chem.MolFromSmiles(smiles_string)

    faces, left, bond2id = getface(mol)
    for i, face in enumerate(faces):
        print(i, *face)
    for bond, idx in bond2id.items():
        print(bond, left[idx])
