import torch
from torch import nn, Tensor
from typing import Optional, Tuple
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
import math
from torch_geometric.utils import softmax
from torch_scatter import scatter


class MetaLayer(nn.Module):
    def __init__(
        self,
        edge_model=None,
        node_model=None,
        face_model=None,
        global_model=None,
        aggregate_edges_for_node_fn=None,
        aggregate_edges_for_globals_fn=None,
        aggregate_nodes_for_globals_fn=None,
        aggregate_edges_for_face_fn=None,
    ):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.face_model = face_model
        self.global_model = global_model
        self.aggregate_edges_for_node_fn = aggregate_edges_for_node_fn
        self.aggregate_edges_for_globals_fn = aggregate_edges_for_globals_fn
        self.aggregate_nodes_for_globals_fn = aggregate_nodes_for_globals_fn
        self.aggregate_edges_for_face_fn = aggregate_edges_for_face_fn

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        u: Optional[Tensor] = None,
        node_batch: Optional[Tensor] = None,
        edge_batch: Optional[Tensor] = None,
        face_batch: Optional[Tensor] = None,
        face: Optional[Tensor] = None,
        face_mask: Optional[Tensor] = None,
        face_index: Optional[Tensor] = None,
        num_nodes: Optional[Tensor] = None,
        num_faces: Optional[Tensor] = None,
        num_edges: Optional[Tensor] = None,
    ):
        row = edge_index[0]
        col = edge_index[1]
        if self.edge_model is not None:
            # need to update edge
            sent_attributes = x[row]
            received_attributes = x[col]

            if face_index is not None:
                inface = face[face_index[0]]
                outface = face[face_index[1]]
            else:
                inface = None
                outface = None

            global_edges = torch.repeat_interleave(u, num_edges, dim=0)
            feat_list = [edge_attr, sent_attributes, received_attributes, global_edges]
            if face_index is not None:
                feat_list.extend([inface, outface])
            concat_feat = torch.cat(feat_list, dim=1)
            edge_attr = self.edge_model(concat_feat)

        if self.node_model is not None:
            sent_attributes = self.aggregate_edges_for_node_fn(edge_attr, row, size=x.size(0))
            received_attributes = self.aggregate_edges_for_node_fn(edge_attr, col, size=x.size(0))
            global_nodes = torch.repeat_interleave(u, num_nodes, dim=0)
            feat_list = [x, sent_attributes, received_attributes, global_nodes]
            x = self.node_model(torch.cat(feat_list, dim=1))

        if self.face_model is not None:
            assert face_index is not None
            sent_attributes = self.aggregate_edges_for_face_fn(
                edge_attr, face_index[0], size=face.size(0)
            )
            received_attributes = self.aggregate_edges_for_face_fn(
                edge_attr, face_index[1], size=face.size(0)
            )
            global_faces = torch.repeat_interleave(u, num_faces, dim=0)
            feat_list = [face, sent_attributes, received_attributes, global_faces]
            face = self.face_model(torch.cat(feat_list, dim=1))
            face = torch.where(face_mask.unsqueeze(1), face.new_zeros((face.shape[0], 1)), face)

        if self.global_model is not None:
            n_graph = u.size(0)

            node_attributes = self.aggregate_nodes_for_globals_fn(x, node_batch, size=n_graph)
            edge_attributes = self.aggregate_edges_for_globals_fn(
                edge_attr, edge_batch, size=n_graph
            )
            feat_list = [u, node_attributes, edge_attributes]
            if face_index is not None:

                face_attributes = self.aggregate_nodes_for_globals_fn(
                    face, face_batch, size=n_graph
                )
                feat_list.append(face_attributes)
            u = self.global_model(torch.cat(feat_list, dim=1))

        return x, edge_attr, u, face


class MetaLayer2(nn.Module):
    def __init__(
        self,
        edge_model=None,
        node_model=None,
        face_model=None,
        global_model=None,
        aggregate_edges_for_node_fn=None,
        aggregate_edges_for_globals_fn=None,
        aggregate_nodes_for_globals_fn=None,
        aggregate_edges_for_face_fn=None,
        residual=False,
        node_attn=False,
        face_attn=False,
        global_attn=False,
        embed_dim=None,
    ):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.face_model = face_model
        self.global_model = global_model
        self.aggregate_edges_for_node_fn = aggregate_edges_for_node_fn
        self.aggregate_edges_for_globals_fn = aggregate_edges_for_globals_fn
        self.aggregate_nodes_for_globals_fn = aggregate_nodes_for_globals_fn
        self.aggregate_edges_for_face_fn = aggregate_edges_for_face_fn
        self.residual = residual

        if node_attn:
            self.node_attn = NodeAttn(embed_dim, num_heads=None)
        else:
            self.node_attn = None

        if face_attn and self.face_model is not None:
            self.face_node_attn = GlobalAttn(embed_dim, num_heads=None)
            self.face_edge_attn = GlobalAttn(embed_dim, num_heads=None)
        else:
            self.face_node_attn = None
            self.face_edge_attn = None

        if global_attn and self.global_model is not None:
            self.global_node_attn = GlobalAttn(embed_dim, num_heads=None)
            self.global_edge_attn = GlobalAttn(embed_dim, num_heads=None)
            self.global_face_attn = GlobalAttn(embed_dim, num_heads=None)
        else:
            self.global_node_attn = None
            self.global_edge_attn = None
            self.global_face_attn = None

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        u: Optional[Tensor] = None,
        node_batch: Optional[Tensor] = None,
        edge_batch: Optional[Tensor] = None,
        face_batch: Optional[Tensor] = None,
        face: Optional[Tensor] = None,
        face_mask: Optional[Tensor] = None,
        face_index: Optional[Tensor] = None,
        num_nodes: Optional[Tensor] = None,
        num_faces: Optional[Tensor] = None,
        num_edges: Optional[Tensor] = None,
        nf_node: Optional[Tensor] = None,
        nf_face: Optional[Tensor] = None,
    ):
        row = edge_index[0]
        col = edge_index[1]
        if self.edge_model is not None:
            # need to update edge
            sent_attributes = x[row]
            received_attributes = x[col]

            if face_index is not None:
                inface = face[face_index[0]]
                outface = face[face_index[1]]
            else:
                inface = None
                outface = None

            global_edges = torch.repeat_interleave(u, num_edges, dim=0)
            feat_list = [edge_attr, sent_attributes, received_attributes, global_edges]
            if face_index is not None:
                feat_list.extend([inface, outface])
            concat_feat = torch.cat(feat_list, dim=1)
            if self.residual:
                edge_attr = edge_attr + self.edge_model(concat_feat)
            else:
                edge_attr = self.edge_model(concat_feat)

        if self.node_model is not None and self.node_attn is None:
            sent_attributes = self.aggregate_edges_for_node_fn(edge_attr, row, size=x.size(0))
            received_attributes = self.aggregate_edges_for_node_fn(edge_attr, col, size=x.size(0))
            global_nodes = torch.repeat_interleave(u, num_nodes, dim=0)
            feat_list = [x, sent_attributes, received_attributes, global_nodes]
            if face_index is not None:
                face_attributes = self.aggregate_edges_for_node_fn(
                    face[nf_face], nf_node, size=x.size(0)
                )
                feat_list.append(face_attributes)
            if self.residual:
                x = x + self.node_model(torch.cat(feat_list, dim=1))
            else:
                x = self.node_model(torch.cat(feat_list, dim=1))
        elif self.node_model is not None:
            max_node_id = x.size(0)
            sent_attributes = self.node_attn(x[row], x[col], edge_attr, row, max_node_id)
            received_attributes = self.node_attn(x[col], x[row], edge_attr, col, max_node_id)
            global_nodes = torch.repeat_interleave(u, num_nodes, dim=0)
            feat_list = [x, sent_attributes, received_attributes, global_nodes]
            if face_index is not None:
                face_attributes = self.aggregate_edges_for_node_fn(
                    face[nf_face], nf_node, size=x.size(0)
                )
                feat_list.append(face_attributes)

            if self.residual:
                x = x + self.node_model(torch.cat(feat_list, dim=1))
            else:
                x = self.node_model(torch.cat(feat_list, dim=1))

        if self.face_model is not None and self.face_node_attn is None:
            assert face_index is not None
            sent_attributes = self.aggregate_edges_for_face_fn(
                edge_attr, face_index[0], size=face.size(0)
            )
            received_attributes = self.aggregate_edges_for_face_fn(
                edge_attr, face_index[1], size=face.size(0)
            )
            global_faces = torch.repeat_interleave(u, num_faces, dim=0)
            node_attributes = self.aggregate_edges_for_face_fn(
                x[nf_node], nf_face, size=face.size(0)
            )
            feat_list = [face, sent_attributes, received_attributes, global_faces, node_attributes]
            residual_face = face
            face = self.face_model(torch.cat(feat_list, dim=1))
            face = torch.where(face_mask.unsqueeze(1), face.new_zeros((face.shape[0], 1)), face)
            if self.residual:
                face = face + residual_face
        elif self.face_model is not None:
            assert face_index is not None
            max_face_id = face.size(0)
            sent_attributes = self.face_edge_attn(
                face[face_index[0]], edge_attr, face_index[0], max_face_id
            )
            received_attributes = self.face_edge_attn(
                face[face_index[1]], edge_attr, face_index[1], max_face_id
            )
            global_faces = torch.repeat_interleave(u, num_faces, dim=0)
            node_attributes = self.face_node_attn(face[nf_face], x[nf_node], nf_face, max_face_id)
            feat_list = [face, sent_attributes, received_attributes, global_faces, node_attributes]
            residual_face = face
            face = self.face_model(torch.cat(feat_list, dim=1))
            face = torch.where(face_mask.unsqueeze(1), face.new_zeros((face.shape[0], 1)), face)
            if self.residual:
                face = face + residual_face

        if self.global_model is not None and self.global_edge_attn is None:
            n_graph = u.size(0)

            node_attributes = self.aggregate_nodes_for_globals_fn(x, node_batch, size=n_graph)
            edge_attributes = self.aggregate_edges_for_globals_fn(
                edge_attr, edge_batch, size=n_graph
            )
            feat_list = [u, node_attributes, edge_attributes]
            if face_index is not None:

                face_attributes = self.aggregate_nodes_for_globals_fn(
                    face, face_batch, size=n_graph
                )
                feat_list.append(face_attributes)
            if self.residual:
                u = u + self.global_model(torch.cat(feat_list, dim=1))
            else:
                u = self.global_model(torch.cat(feat_list, dim=1))
        elif self.global_model is not None:
            n_graph = u.size(0)
            node_attributes = self.global_node_attn(
                torch.repeat_interleave(u, num_nodes, dim=0), x, node_batch, n_graph
            )
            edge_attributes = self.global_edge_attn(
                torch.repeat_interleave(u, num_edges, dim=0), edge_attr, edge_batch, n_graph
            )
            feat_list = [u, node_attributes, edge_attributes]
            if face_index is not None:
                face_attributes = self.global_face_attn(
                    torch.repeat_interleave(u, num_faces, dim=0), face, face_batch, n_graph
                )
                feat_list.append(face_attributes)
            if self.residual:
                u = u + self.global_model(torch.cat(feat_list, dim=1))
            else:
                u = self.global_model(torch.cat(feat_list, dim=1))

        return x, edge_attr, u, face


class MetaLayer3(nn.Module):
    def __init__(
        self,
        edge_model,
        node_model,
        face_model,
        global_model,
        aggregate_edges_for_node_fn=None,
        aggregate_edges_for_globals_fn=None,
        aggregate_nodes_for_globals_fn=None,
        aggregate_edges_for_face_fn=None,
        residual=False,
    ):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.face_model = face_model
        self.global_model = global_model
        self.aggregate_edges_for_node_fn = aggregate_edges_for_node_fn
        self.aggregate_edges_for_globals_fn = aggregate_edges_for_globals_fn
        self.aggregate_nodes_for_globals_fn = aggregate_nodes_for_globals_fn
        self.aggregate_edges_for_face_fn = aggregate_edges_for_face_fn
        self.residual = residual
        assert self.residual == False

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        u: Optional[Tensor] = None,
        node_batch: Optional[Tensor] = None,
        edge_batch: Optional[Tensor] = None,
        face_batch: Optional[Tensor] = None,
        face: Optional[Tensor] = None,
        face_mask: Optional[Tensor] = None,
        face_index: Optional[Tensor] = None,
        num_nodes: Optional[Tensor] = None,
        num_faces: Optional[Tensor] = None,
        num_edges: Optional[Tensor] = None,
        nf_node: Optional[Tensor] = None,
        nf_face: Optional[Tensor] = None,
    ):
        x_residual = x
        edge_attr_residual = edge_attr
        face_residual = face

        row = edge_index[0]
        col = edge_index[1]
        if self.edge_model is not None:
            # need to update edge
            sent_attributes = x[row]
            received_attributes = x[col]

            if face_index is not None:
                inface = face[face_index[0]]
                outface = face[face_index[1]]
            else:
                inface = None
                outface = None

            global_edges = torch.repeat_interleave(u, num_edges, dim=0)
            feat_list = [edge_attr, sent_attributes, received_attributes, global_edges]
            if face_index is not None:
                feat_list.extend([inface, outface])
            concat_feat = torch.cat(feat_list, dim=1)
            if self.residual:
                edge_attr = edge_attr + self.edge_model(concat_feat)
            else:
                edge_attr = self.edge_model(concat_feat)

        if self.node_model is not None:
            sent_attributes = self.aggregate_edges_for_node_fn(
                edge_attr_residual, row, size=x.size(0)
            )
            received_attributes = self.aggregate_edges_for_node_fn(
                edge_attr_residual, col, size=x.size(0)
            )
            global_nodes = torch.repeat_interleave(u, num_nodes, dim=0)
            feat_list = [x, sent_attributes, received_attributes, global_nodes]
            if face_index is not None:
                face_attributes = self.aggregate_edges_for_node_fn(
                    face[nf_face], nf_node, size=x.size(0)
                )
                feat_list.append(face_attributes)
            if self.residual:
                x = x + self.node_model(torch.cat(feat_list, dim=1))
            else:
                x = self.node_model(torch.cat(feat_list, dim=1))

        if self.face_model is not None:
            assert face_index is not None
            sent_attributes = self.aggregate_edges_for_face_fn(
                edge_attr_residual, face_index[0], size=face.size(0)
            )
            received_attributes = self.aggregate_edges_for_face_fn(
                edge_attr_residual, face_index[1], size=face.size(0)
            )
            global_faces = torch.repeat_interleave(u, num_faces, dim=0)
            node_attributes = self.aggregate_edges_for_face_fn(
                x_residual[nf_node], nf_face, size=face.size(0)
            )
            feat_list = [face, sent_attributes, received_attributes, global_faces, node_attributes]
            residual_face = face
            face = self.face_model(torch.cat(feat_list, dim=1))
            face = torch.where(face_mask.unsqueeze(1), face.new_zeros((face.shape[0], 1)), face)
            if self.residual:
                face = face + residual_face

        if self.global_model is not None:
            n_graph = u.size(0)

            node_attributes = self.aggregate_nodes_for_globals_fn(
                x_residual, node_batch, size=n_graph
            )
            edge_attributes = self.aggregate_edges_for_globals_fn(
                edge_attr_residual, edge_batch, size=n_graph
            )
            feat_list = [u, node_attributes, edge_attributes]
            if face_index is not None:

                face_attributes = self.aggregate_nodes_for_globals_fn(
                    face_residual, face_batch, size=n_graph
                )
                feat_list.append(face_attributes)
            if self.residual:
                u = u + self.global_model(torch.cat(feat_list, dim=1))
            else:
                u = self.global_model(torch.cat(feat_list, dim=1))

        return x, edge_attr, u, face


class DropoutIfTraining(nn.Module):
    """
    Borrow this implementation from deepmind
    """

    def __init__(self, p=0.0, submodule=None):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = p
        self.submodule = submodule if submodule else nn.Identity()

    def forward(self, x):
        x = self.submodule(x)
        newones = x.new_ones((x.size(0), 1))
        newones = F.dropout(newones, p=self.p, training=self.training)
        out = x * newones
        return out


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_sizes,
        use_layer_norm=False,
        activation=nn.ReLU,
        dropout=0.0,
        layernorm_before=False,
        use_bn=False,
    ):
        super().__init__()
        module_list = []
        if not use_bn:
            if layernorm_before:
                module_list.append(nn.LayerNorm(input_size))
            for size in output_sizes:
                module_list.append(nn.Linear(input_size, size))
                if size != 1:
                    module_list.append(activation())
                    if dropout > 0:
                        module_list.append(nn.Dropout(dropout))
                input_size = size
            if not layernorm_before and use_layer_norm:
                module_list.append(nn.LayerNorm(input_size))
        else:
            for size in output_sizes:
                module_list.append(nn.Linear(input_size, size))
                if size != 1:
                    module_list.append(nn.BatchNorm1d(size))
                    module_list.append(activation())
                    if dropout > 0:
                        module_list.append(nn.Dropout(p=dropout))
                input_size = size

        self.module_list = nn.ModuleList(module_list)
        self.reset_parameters()

    def reset_parameters(self):
        for item in self.module_list:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(self, x):
        for item in self.module_list:
            x = item(x)
        return x


class MLPwoLastAct(nn.Module):
    def __init__(
        self,
        input_size,
        output_sizes,
        use_layer_norm=False,
        activation=nn.ReLU,
        dropout=0.0,
        layernorm_before=False,
        use_bn=False,
    ):
        super().__init__()
        module_list = []
        if not use_bn:
            if layernorm_before:
                module_list.append(nn.LayerNorm(input_size))

            for i, size in enumerate(output_sizes):
                module_list.append(nn.Linear(input_size, size))
                if i < len(output_sizes) - 1:
                    module_list.append(activation())
                    if dropout > 0:
                        module_list.append(nn.Dropout(dropout))
                input_size = size

            if not layernorm_before and use_layer_norm:
                module_list.append(nn.LayerNorm(input_size))
        else:
            for i, size in enumerate(output_sizes):
                module_list.append(nn.Linear(input_size, size))
                if i < len(output_sizes) - 1:
                    module_list.append(nn.BatchNorm1d(size))
                    module_list.append(activation())
                    if dropout > 0:
                        module_list.append(nn.Dropout(p=dropout))
                input_size = size

        self.module_list = nn.ModuleList(module_list)
        self.reset_parameters()

    def reset_parameters(self):
        for item in self.module_list:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(self, x):
        for item in self.module_list:
            x = item(x)
        return x


class AtomEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.m = nn.Linear(sum(get_atom_feature_dims()), emb_dim)

    def forward(self, x):
        return self.m(x)


class BondEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.m = nn.Linear(sum(get_bond_feature_dims()), emb_dim)

    def forward(self, x):
        return self.m(x)


class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.eps = nn.Parameter(torch.Tensor([0.0]))
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp(
            (1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        )
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class NodeAttn(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.emb_dim = embed_dim
        if num_heads is None:
            num_heads = embed_dim // 64
        self.num_heads = num_heads
        assert self.emb_dim % self.num_heads == 0

        self.w1 = nn.Linear(3 * embed_dim, embed_dim)
        self.w2 = nn.Parameter(torch.zeros((self.num_heads, self.emb_dim // self.num_heads)))
        self.w3 = nn.Linear(2 * embed_dim, embed_dim)
        self.head_dim = embed_dim // self.num_heads
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w2, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w3.weight, gain=1 / math.sqrt(2))

    def forward(self, q, k_v, k_e, index, nnode):

        x = torch.cat([q, k_v, k_e], dim=-1)
        x = self.w1(x).view(-1, self.num_heads, self.head_dim)
        x = F.leaky_relu(x)
        attn_weight = torch.einsum("nhc,hc->nh", x, self.w2).unsqueeze(-1)
        attn_weight = softmax(attn_weight, index)
        v = torch.cat([k_v, k_e], dim=1)
        v = self.w3(v).view(-1, self.num_heads, self.head_dim)
        x = (attn_weight * v).reshape(-1, self.emb_dim)
        x = scatter(x, index, dim=0, reduce="sum", dim_size=nnode)
        return x


class GlobalAttn(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.emb_dim = emb_dim
        if num_heads is None:
            num_heads = emb_dim // 64
        self.num_heads = num_heads
        assert self.emb_dim % self.num_heads == 0
        self.w1 = nn.Linear(2 * emb_dim, emb_dim)
        self.w2 = nn.Parameter(torch.zeros(self.num_heads, self.emb_dim // self.num_heads))
        self.head_dim = self.emb_dim // self.num_heads
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.w1.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w2, gain=1 / math.sqrt(2))

    def forward(self, q, k, index, dim_size):
        x = torch.cat([q, k], dim=1)
        x = self.w1(x).view(-1, self.num_heads, self.head_dim)
        x = F.leaky_relu(x)
        attn_weight = torch.einsum("nhc,hc->nh", x, self.w2).unsqueeze(-1)
        attn_weight = softmax(attn_weight, index)

        v = k.view(-1, self.num_heads, self.head_dim)
        x = (attn_weight * v).reshape(-1, self.emb_dim)
        x = scatter(x, index, dim=0, reduce="sum", dim_size=dim_size)
        return x

