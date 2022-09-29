import torch
from torch import nn
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    GlobalAttention,
    Set2Set,
)
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from dualgraph.conv import (
    MLP,
    DropoutIfTraining,
    MetaLayer,
    AtomEncoder,
    GINConv,
    MetaLayer2,
    MetaLayer3,
    MLPwoLastAct,
)
import torch.nn.functional as F
from dualgraph.utils import GradMultiply

_REDUCER_NAMES = {"sum": global_add_pool, "mean": global_mean_pool, "max": global_max_pool}


class GNN(nn.Module):
    def __init__(
        self,
        mlp_hidden_size: int = 512,
        mlp_layers: int = 2,
        latent_size: int = 128,
        use_layer_norm: bool = False,
        num_message_passing_steps: int = 8,
        global_reducer: str = "sum",
        node_reducer: str = "sum",
        face_reducer: str = "sum",
        dropedge_rate: float = 0.1,
        dropnode_rate: float = 0.1,
        ignore_globals: bool = True,
        use_face: bool = True,
        dropout: float = 0.1,
        dropnet: float = 0.1,
        init_face: bool = False,
        graph_pooling: str = "sum",
        use_outer: bool = False,
    ):
        super().__init__()

        self.encoder_edge = MLP(
            sum(get_bond_feature_dims()),
            [mlp_hidden_size] * mlp_layers + [latent_size],
            use_layer_norm=use_layer_norm,
        )
        self.encoder_node = MLP(
            sum(get_atom_feature_dims()),
            [mlp_hidden_size] * mlp_layers + [latent_size],
            use_layer_norm=use_layer_norm,
        )
        self.encoder_global = MLP(
            latent_size,
            [mlp_hidden_size] * mlp_layers + [latent_size],
            use_layer_norm=use_layer_norm,
        )
        if use_face:
            self.encoder_face = MLP(
                latent_size * (2 if init_face else 1),
                [mlp_hidden_size] * mlp_layers + [latent_size],
                use_layer_norm=use_layer_norm,
            )
        else:
            self.encoder_face = None

        self.gnn_layers = nn.ModuleList()
        for _ in range(num_message_passing_steps):
            edge_model = DropoutIfTraining(
                p=dropedge_rate,
                submodule=MLP(
                    latent_size * (6 if use_face else 4),
                    [mlp_hidden_size] * mlp_layers + [latent_size],
                    use_layer_norm=use_layer_norm,
                ),
            )
            node_model = DropoutIfTraining(
                p=dropnode_rate,
                submodule=MLP(
                    latent_size * 4,
                    [mlp_hidden_size] * mlp_layers + [latent_size],
                    use_layer_norm=use_layer_norm,
                ),
            )
            global_model = MLP(
                latent_size * (4 if use_face else 3),
                [mlp_hidden_size] * mlp_layers + [latent_size],
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
            if use_face:
                face_model = MLP(
                    latent_size * 4,
                    [mlp_hidden_size] * mlp_layers + [latent_size],
                    use_layer_norm=use_layer_norm,
                    dropout=dropout,
                )
            else:
                face_model = None
            self.gnn_layers.append(
                MetaLayer(
                    edge_model=edge_model,
                    node_model=node_model,
                    face_model=face_model,
                    global_model=global_model,
                    aggregate_edges_for_node_fn=_REDUCER_NAMES[node_reducer],
                    aggregate_edges_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    aggregate_nodes_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    aggregate_edges_for_face_fn=_REDUCER_NAMES[face_reducer],
                )
            )

        decoder_input_dim = latent_size
        if use_face:
            decoder_input_dim += latent_size
        if not ignore_globals:
            decoder_input_dim += latent_size

        self.decoder = MLP(
            decoder_input_dim, [mlp_hidden_size] * mlp_layers + [1], use_layer_norm=False
        )

        self.use_face = use_face
        self.latent_size = latent_size
        self.aggregate_nodes_for_globals_fn = _REDUCER_NAMES[global_reducer]
        self.aggregate_edges_for_node_fn = _REDUCER_NAMES[node_reducer]
        self.aggregate_edges_for_face_fn = _REDUCER_NAMES[face_reducer]
        self.ignore_globals = ignore_globals
        self.pooling = _REDUCER_NAMES[graph_pooling]

        self.dropnet = dropnet
        self.init_face = init_face
        self.use_outer = use_outer

    def forward(self, batch):
        (
            x,
            edge_index,
            edge_attr,
            node_batch,
            face_mask,
            face_index,
            num_nodes,
            num_faces,
            num_edges,
            num_graphs,
        ) = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.ring_mask,
            batch.ring_index,
            batch.n_nodes,
            batch.num_rings,
            batch.n_edges,
            batch.num_graphs,
        )

        x = one_hot_atoms(x)
        edge_attr = one_hot_bonds(edge_attr)

        graph_idx = torch.arange(num_graphs).to(x.device)
        edge_batch = torch.repeat_interleave(graph_idx, num_edges, dim=0)

        u = x.new_zeros((num_graphs, self.latent_size))

        x = self.encoder_node(x)
        edge_attr = self.encoder_edge(edge_attr)
        u = self.encoder_global(u)

        if self.use_face:
            face_batch = torch.repeat_interleave(graph_idx, num_faces, dim=0)
            if self.init_face:
                sent_attributes = x[edge_index[0]]
                received_attributes = x[edge_index[1]]
                feat = torch.cat([edge_attr, sent_attributes + received_attributes], dim=1)
                face = self.aggregate_edges_for_face_fn(
                    feat, face_index[0], size=num_faces.sum().item()
                )
                face = self.encoder_face(face)
                face = torch.where(face_mask.unsqueeze(1), face.new_zeros((face.shape[0], 1)), face)
            else:
                face = x.new_zeros((num_faces.sum().item(), self.latent_size))
                face = self.encoder_face(face)
        else:
            face = None
            face_batch = None
            face_index = None

        for layer in self.gnn_layers:
            x_1, edge_attr_1, u_1, face_1 = layer(
                x,
                edge_index,
                edge_attr,
                u,
                node_batch,
                edge_batch,
                face_batch,
                face,
                face_mask,
                face_index,
                num_nodes,
                num_faces,
                num_edges,
            )
            x = x_1 + x
            edge_attr = edge_attr_1 + edge_attr
            u = u_1 + u
            if face is not None:
                face = face_1 + face

        aggregated_feats = []
        aggregated_feats.append(self.pooling(x, node_batch, size=num_graphs))
        if self.use_face:
            aggregated_feats.append(self.pooling(face, face_batch, size=num_graphs))
            if self.dropnet > 0:
                random_1 = (
                    aggregated_feats[0].new_zeros((aggregated_feats[0].shape[0], 1)).uniform_()
                )
                random_2 = 1 - random_1
                mask_2 = (random_2 >= self.dropnet).to(torch.float32)
                mask_1 = ((random_1 >= self.dropnet) | (num_faces.unsqueeze(1).eq(1))).to(
                    torch.float32
                )
                aggregated_feats = [aggregated_feats[0] * mask_1, aggregated_feats[1] * mask_2]
        if not self.ignore_globals:
            aggregated_feats.append(
                u * (u.new_zeros((u.shape[0], 1)).uniform_() >= self.dropnet).to(torch.float32)
            )
        aggregated_feat = torch.cat(aggregated_feats, dim=1)
        out = self.decoder(aggregated_feat)
        return out


class GNN2(nn.Module):
    def __init__(
        self,
        mlp_hidden_size: int = 512,
        mlp_layers: int = 2,
        latent_size: int = 128,
        use_layer_norm: bool = False,
        num_message_passing_steps: int = 8,
        global_reducer: str = "sum",
        node_reducer: str = "sum",
        face_reducer: str = "sum",
        dropedge_rate: float = 0.1,
        dropnode_rate: float = 0.1,
        ignore_globals: bool = True,
        use_face: bool = True,
        dropout: float = 0.1,
        dropnet: float = 0.1,
        init_face: bool = False,
        graph_pooling: str = "sum",
        use_outer: bool = False,
        residual: bool = False,
        layernorm_before: bool = False,
        parallel: bool = False,
        num_tasks: int = 1,
        layer_drop: float = 0.0,
        pooler_dropout: float = 0.0,
        encoder_dropout: float = 0.0,
        use_bn: bool = False,
        global_attn: bool = False,
        node_attn: bool = False,
        face_attn: bool = False,
        flag: bool = False,
        ddi: bool = False,
        gradmultiply: float = -1,
        ap_hid_size: int = None,
        ap_mlp_layers: int = None,
    ):
        super().__init__()

        if flag:
            self.encoder_edge = nn.Linear(sum(get_bond_feature_dims()), latent_size, bias=False)
            self.encoder_node = nn.Linear(sum(get_atom_feature_dims()), latent_size, bias=False)
        else:
            self.encoder_edge = MLPwoLastAct(
                sum(get_bond_feature_dims()),
                [mlp_hidden_size] * mlp_layers + [latent_size],
                use_layer_norm=use_layer_norm,
                use_bn=use_bn,
            )
            self.encoder_node = MLPwoLastAct(
                sum(get_atom_feature_dims()),
                [mlp_hidden_size] * mlp_layers + [latent_size],
                use_layer_norm=use_layer_norm,
                use_bn=use_bn,
            )
        self.global_init = nn.Parameter(torch.zeros((1, latent_size)))
        if use_face:
            self.encoder_face = MLPwoLastAct(
                latent_size * (3 if init_face else 1),
                [mlp_hidden_size] * mlp_layers + [latent_size],
                use_layer_norm=use_layer_norm,
                use_bn=use_bn,
                layernorm_before=layernorm_before,
            )
        else:
            self.encoder_face = None

        self.gnn_layers = nn.ModuleList()
        for i in range(num_message_passing_steps):
            edge_model = DropoutIfTraining(
                p=dropedge_rate,
                submodule=MLP(
                    latent_size * (6 if use_face else 4),
                    [mlp_hidden_size] * mlp_layers + [latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=encoder_dropout,
                    use_bn=use_bn,
                ),
            )
            node_model = DropoutIfTraining(
                p=dropnode_rate,
                submodule=MLP(
                    latent_size * (5 if use_face else 4),
                    [mlp_hidden_size] * mlp_layers + [latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=encoder_dropout,
                    use_bn=use_bn,
                ),
            )
            if i == num_message_passing_steps - 1:
                global_model = None
            else:

                global_model = MLP(
                    latent_size * (4 if use_face else 3),
                    [mlp_hidden_size] * mlp_layers + [latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=encoder_dropout,
                    use_bn=use_bn,
                )
            if use_face:
                if i == num_message_passing_steps - 1:
                    face_model = None
                else:
                    face_model = MLP(
                        latent_size * 5,
                        [mlp_hidden_size] * mlp_layers + [latent_size],
                        use_layer_norm=use_layer_norm,
                        layernorm_before=layernorm_before,
                        dropout=encoder_dropout,
                        use_bn=use_bn,
                    )
            else:
                face_model = None
            sublayer = MetaLayer3 if parallel else MetaLayer2
            self.gnn_layers.append(
                sublayer(
                    edge_model=edge_model,
                    node_model=node_model,
                    face_model=face_model,
                    global_model=global_model,
                    aggregate_edges_for_node_fn=_REDUCER_NAMES[node_reducer],
                    aggregate_edges_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    aggregate_nodes_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    aggregate_edges_for_face_fn=_REDUCER_NAMES[face_reducer],
                    face_attn=face_attn,
                    node_attn=node_attn,
                    global_attn=global_attn,
                    embed_dim=latent_size,
                )
            )
        self.ddi = ddi
        ap_hid_size = mlp_hidden_size if ap_hid_size is None else ap_hid_size
        ap_mlp_layers = mlp_layers if ap_mlp_layers is None else ap_mlp_layers
        self.decoder = MLPwoLastAct(
            latent_size if not self.ddi else 2 * latent_size,
            [ap_hid_size] * ap_mlp_layers + [num_tasks],
            use_layer_norm=False,
            dropout=pooler_dropout,
            use_bn=use_bn,
        )

        self.use_face = use_face
        self.latent_size = latent_size
        self.aggregate_nodes_for_globals_fn = _REDUCER_NAMES[global_reducer]
        self.aggregate_edges_for_node_fn = _REDUCER_NAMES[node_reducer]
        self.aggregate_edges_for_face_fn = _REDUCER_NAMES[face_reducer]
        self.ignore_globals = ignore_globals
        self.pooling = _REDUCER_NAMES[graph_pooling]

        self.dropnet = dropnet
        self.init_face = init_face
        self.use_outer = use_outer
        self.residual = residual
        self.layer_drop = layer_drop
        self.dropout = dropout
        self.reset_parameters()
        self.gradmultiply = gradmultiply

    def reset_parameters(self):
        if isinstance(self.encoder_node, nn.Linear):
            nn.init.xavier_uniform_(self.encoder_node.weight.data)
            nn.init.xavier_uniform_(self.encoder_edge.weight.data)

    def forward(self, batch, perturb=None, perturb_edge=None):
        (
            x,
            edge_index,
            edge_attr,
            node_batch,
            face_mask,
            face_index,
            num_nodes,
            num_faces,
            num_edges,
            num_graphs,
            nf_node,
            nf_face,
        ) = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.ring_mask,
            batch.ring_index,
            batch.n_nodes,
            batch.num_rings,
            batch.n_edges,
            batch.num_graphs,
            batch.nf_node.view(-1),
            batch.nf_ring.view(-1),
        )

        if self.use_outer:
            face_mask = face_mask.fill_(False)
        x = one_hot_atoms(x)
        edge_attr = one_hot_bonds(edge_attr)

        graph_idx = torch.arange(num_graphs).to(x.device)
        edge_batch = torch.repeat_interleave(graph_idx, num_edges, dim=0)

        u = self.global_init.expand(num_graphs, -1)

        x = self.encoder_node(x)
        if perturb is not None:
            x = x + perturb
        edge_attr = self.encoder_edge(edge_attr)
        if perturb_edge is not None:
            edge_attr = edge_attr + perturb_edge

        if self.use_face:
            face_batch = torch.repeat_interleave(graph_idx, num_faces, dim=0)
            if self.init_face:
                node_attributes = self.aggregate_edges_for_face_fn(
                    x[nf_node], nf_face, size=num_faces.sum().item()
                )
                sent_attributes = self.aggregate_edges_for_face_fn(
                    edge_attr, face_index[0], size=num_faces.sum().item()
                )
                received_attributes = self.aggregate_edges_for_face_fn(
                    edge_attr, face_index[1], size=num_faces.sum().item()
                )
                feat = torch.cat([node_attributes, sent_attributes, received_attributes], dim=1)
                feat = torch.where(face_mask.unsqueeze(1), feat.new_zeros((feat.shape[0], 1)), feat)
                face = self.encoder_face(feat)
                # face = torch.where(face_mask.unsqueeze(1), face.new_zeros((face.shape[0], 1)), face)
            else:
                face = x.new_zeros((num_faces.sum().item(), self.latent_size))
                face = self.encoder_face(face)
        else:
            face = None
            face_batch = None
            face_index = None

        droplayer_probs = torch.empty(len(self.gnn_layers)).uniform_()
        for i, layer in enumerate(self.gnn_layers):
            if self.training and self.layer_drop > 0 and droplayer_probs[i] < self.layer_drop:
                continue
            x_1, edge_attr_1, u_1, face_1 = layer(
                x,
                edge_index,
                edge_attr,
                u,
                node_batch,
                edge_batch,
                face_batch,
                face,
                face_mask,
                face_index,
                num_nodes,
                num_faces,
                num_edges,
                nf_node,
                nf_face,
            )
            if self.residual:
                x = x_1
                edge_attr = edge_attr_1
                u = u_1
                if face is not None:
                    face = face_1
            else:
                x = F.dropout(x_1, p=self.dropout, training=self.training) + x
                edge_attr = (
                    F.dropout(edge_attr_1, p=self.dropout, training=self.training) + edge_attr
                )
                u = F.dropout(u_1, p=self.dropout, training=self.training) + u
                if face is not None:
                    face = F.dropout(face_1, p=self.dropout, training=self.training) + face

        if self.ddi:
            return self.pooling(x, node_batch, size=num_graphs)

        if self.gradmultiply > 0:
            x = self.pooling(x, node_batch, size=num_graphs)
            x = GradMultiply.apply(x, self.gradmultiply)
            out = self.decoder(x)
        else:
            out = self.decoder(self.pooling(x, node_batch, size=num_graphs))
        return out

    def get_last_layer(self, input):
        assert self.ddi
        return self.decoder(input)


class GNNwithvn(GNN2):
    def __init__(
        self, latent_size: int = 128, num_message_passing_steps: int = 8, **kwargs,
    ):
        super().__init__(
            latent_size=latent_size, num_message_passing_steps=num_message_passing_steps, **kwargs
        )
        self.virtualnode_embedding = torch.nn.Embedding(1, latent_size)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        self.mlp_virtualnode_list = torch.nn.ModuleList()
        for _ in range(num_message_passing_steps - 1):
            self.mlp_virtualnode_list.append(
                nn.Sequential(
                    nn.Linear(latent_size, 2 * latent_size),
                    nn.BatchNorm1d(2 * latent_size),
                    nn.ReLU(),
                    nn.Linear(2 * latent_size, latent_size),
                    nn.BatchNorm1d(latent_size),
                    nn.ReLU(),
                )
            )

        self.num_layers = num_message_passing_steps

    def forward(self, batch):
        (
            x,
            edge_index,
            edge_attr,
            node_batch,
            face_mask,
            face_index,
            num_nodes,
            num_faces,
            num_edges,
            num_graphs,
            nf_node,
            nf_face,
        ) = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.ring_mask,
            batch.ring_index,
            batch.n_nodes,
            batch.num_rings,
            batch.n_edges,
            batch.num_graphs,
            batch.nf_node.view(-1),
            batch.nf_ring.view(-1),
        )

        if self.use_outer:
            face_mask = face_mask.fill_(False)
        x = one_hot_atoms(x)
        edge_attr = one_hot_bonds(edge_attr)

        graph_idx = torch.arange(num_graphs).to(x.device)
        edge_batch = torch.repeat_interleave(graph_idx, num_edges, dim=0)

        u = x.new_zeros((num_graphs, self.latent_size))

        x = self.encoder_node(x)
        edge_attr = self.encoder_edge(edge_attr)
        u = self.encoder_global(u)

        if self.use_face:
            face_batch = torch.repeat_interleave(graph_idx, num_faces, dim=0)
            if self.init_face:
                node_attributes = self.aggregate_edges_for_face_fn(
                    x[nf_node], nf_face, size=num_faces.sum().item()
                )
                sent_attributes = self.aggregate_edges_for_face_fn(
                    edge_attr, face_index[0], size=num_faces.sum().item()
                )
                received_attributes = self.aggregate_edges_for_face_fn(
                    edge_attr, face_index[1], size=num_faces.sum().item()
                )
                feat = torch.cat([node_attributes, sent_attributes, received_attributes], dim=1)
                feat = torch.where(face_mask.unsqueeze(1), feat.new_zeros((feat.shape[0], 1)), feat)
                face = self.encoder_face(feat)
                # face = torch.where(face_mask.unsqueeze(1), face.new_zeros((face.shape[0], 1)), face)
            else:
                face = x.new_zeros((num_faces.sum().item(), self.latent_size))
                face = self.encoder_face(face)
        else:
            face = None
            face_batch = None
            face_index = None

        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(node_batch[-1].item() + 1).to(edge_index)
        )
        droplayer_probs = torch.empty(len(self.gnn_layers)).uniform_()

        for i, layer in enumerate(self.gnn_layers):
            if self.training and self.layer_drop > 0 and droplayer_probs[i] < self.layer_drop:
                continue
            x = x + virtualnode_embedding[node_batch]
            x_1, edge_attr_1, u_1, face_1 = layer(
                x,
                edge_index,
                edge_attr,
                u,
                node_batch,
                edge_batch,
                face_batch,
                face,
                face_mask,
                face_index,
                num_nodes,
                num_faces,
                num_edges,
                nf_node,
                nf_face,
            )
            if self.residual:
                x = x_1
                edge_attr = edge_attr_1
                u = u_1
                if face is not None:
                    face = face_1
            else:
                x = F.dropout(x_1, p=self.dropout, training=self.training) + x
                edge_attr = (
                    F.dropout(edge_attr_1, p=self.dropout, training=self.training) + edge_attr
                )
                u = F.dropout(u_1, p=self.dropout, training=self.training) + u
                if face is not None:
                    face = F.dropout(face_1, p=self.dropout, training=self.training) + face
            if i < self.num_layers - 1:
                virtualnode_embedding_temp = global_add_pool(x, node_batch) + virtualnode_embedding
                virtualnode_embedding = virtualnode_embedding + F.dropout(
                    self.mlp_virtualnode_list[i](virtualnode_embedding_temp),
                    self.dropout,
                    training=self.training,
                )

        aggregated_feats = []
        aggregated_feats.append(self.pooling(x, node_batch, size=num_graphs))
        if self.use_face:
            aggregated_feats.append(self.pooling(face, face_batch, size=num_graphs))
            if self.dropnet > 0:
                random_1 = (
                    aggregated_feats[0].new_zeros((aggregated_feats[0].shape[0], 1)).uniform_()
                )
                random_2 = 1 - random_1
                mask_2 = (random_2 >= self.dropnet).to(torch.float32)
                mask_1 = ((random_1 >= self.dropnet) | (num_faces.unsqueeze(1).eq(1))).to(
                    torch.float32
                )
                aggregated_feats = [aggregated_feats[0] * mask_1, aggregated_feats[1] * mask_2]
        if not self.ignore_globals:
            aggregated_feats.append(
                u * (u.new_zeros((u.shape[0], 1)).uniform_() >= self.dropnet).to(torch.float32)
            )
        aggregated_feat = torch.cat(aggregated_feats, dim=1)
        out = self.decoder(aggregated_feat)
        return out


def one_hot_atoms(atoms):
    vocab_sizes = get_atom_feature_dims()
    one_hots = []
    for i in range(atoms.shape[1]):
        one_hots.append(
            F.one_hot(atoms[:, i], num_classes=vocab_sizes[i]).to(atoms.device).to(torch.float32)
        )
    return torch.cat(one_hots, dim=1)


def one_hot_bonds(bonds):
    vocab_sizes = get_bond_feature_dims()
    one_hots = []
    for i in range(bonds.shape[1]):
        one_hots.append(
            F.one_hot(bonds[:, i], num_classes=vocab_sizes[i]).to(bonds.device).to(torch.float32)
        )
    return torch.cat(one_hots, dim=1)


class GIN_node_Virtual(nn.Module):
    def __init__(
        self,
        num_layers,
        emb_dim=512,
        dropout=0.1,
        residual=False,
        use_face=True,
        graph_pooling="mean",
        init_face=False,
        dropnet=0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.mlp_virtualnode_list = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GINConv(emb_dim))
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        for _ in range(num_layers - 1):
            self.mlp_virtualnode_list.append(
                nn.Sequential(
                    nn.Linear(emb_dim * (2 if use_face else 1), emb_dim),
                    nn.BatchNorm1d(emb_dim),
                    nn.ReLU(),
                    nn.Linear(emb_dim, emb_dim),
                    nn.BatchNorm1d(emb_dim),
                    nn.ReLU(),
                )
            )
        self.use_face = use_face
        self.init_face = init_face
        self.dropnet = dropnet

        if use_face:
            self.face_encoder = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.BatchNorm1d(emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim),
            )
            self.convs_face = nn.ModuleList()
            self.batch_norms_face = nn.ModuleList()
            for i in range(num_layers):
                conv = GINConv(emb_dim)
                conv.bond_encoder = self.convs[i].bond_encoder
                self.convs_face.append(conv)
                self.batch_norms_face.append(nn.BatchNorm1d(emb_dim))

        self.pool = _REDUCER_NAMES[graph_pooling]
        self.graph_pred_linear = nn.Sequential(
            nn.Linear(emb_dim * (2 if self.use_face else 1), emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1),
        )

        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)

    def forward(self, batch):
        (
            x,
            edge_index,
            edge_attr,
            node_batch,
            face_mask,
            face_index,
            num_nodes,
            num_faces,
            num_edges,
            num_graphs,
        ) = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.ring_mask,
            batch.ring_index,
            batch.n_nodes,
            batch.num_rings,
            batch.n_edges,
            batch.num_graphs,
        )
        x = one_hot_atoms(x)
        edge_attr = one_hot_bonds(edge_attr)

        virtualnode_embedding = self.virtualnode_embedding(edge_index.new_zeros((num_graphs)))

        x = self.atom_encoder(x)

        if self.use_face:
            graph_idx = torch.arange(num_graphs).to(x.device)
            face_batch = torch.repeat_interleave(graph_idx, num_faces, dim=0)
            if self.init_face:
                raise NotImplementedError()
            else:
                face = x.new_zeros((num_faces.sum().item(), self.emb_dim))
                face = self.face_encoder(face)

        for layer in range(self.num_layers):
            h = x + virtualnode_embedding[node_batch]
            h = self.convs[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)

            if self.residual:
                x = h + x
            else:
                x = h

            if self.use_face:
                h = face + virtualnode_embedding[face_batch]
                h = self.convs_face[layer](h, face_index, edge_attr)
                h = self.batch_norms_face[layer](h)
                if layer == self.num_layers - 1:
                    h = F.dropout(h, self.dropout, training=self.training)
                else:
                    h = F.dropout(F.relu(h), self.dropout, training=self.training)
                if self.residual:
                    face = h + face
                else:
                    face = h
                face = torch.where(face_mask.unsqueeze(1), face.new_zeros((face.shape[0], 1)), face)

            if layer < self.num_layers - 1:
                aggregated_feats = []
                aggregated_feats.append(
                    global_add_pool(x, node_batch, size=num_graphs) + virtualnode_embedding
                )
                if self.use_face:
                    aggregated_feats.append(
                        global_add_pool(face, face_batch, size=num_graphs) + virtualnode_embedding
                    )
                    if self.dropnet > 0:
                        random_1 = (
                            aggregated_feats[0]
                            .new_zeros((aggregated_feats[0].shape[0], 1))
                            .uniform_()
                        )
                        random_2 = 1 - random_1
                        mask_2 = (random_2 > self.dropnet).to(torch.float32)
                        mask_1 = ((random_1 > self.dropnet) | (num_faces.unsqueeze(1).eq(1))).to(
                            torch.float32
                        )
                        aggregated_feats = [
                            aggregated_feats[0] * mask_1,
                            aggregated_feats[1] * mask_2,
                        ]
                aggregated_feat = torch.cat(aggregated_feats, dim=1)
                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        self.mlp_virtualnode_list[layer](aggregated_feat),
                        self.dropout,
                        training=self.training,
                    )
                else:
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer](aggregated_feat),
                        self.dropout,
                        training=self.training,
                    )
        aggregated_feats = []
        aggregated_feats.append(self.pool(x, node_batch, size=num_graphs))
        if self.use_face:
            aggregated_feats.append(self.pool(face, face_batch, size=num_graphs))
            if self.dropnet > 0:
                random_1 = (
                    aggregated_feats[0].new_zeros((aggregated_feats[0].shape[0], 1)).uniform_()
                )
                random_2 = 1 - random_1
                mask_2 = (random_2 > self.dropnet).to(torch.float32)
                mask_1 = ((random_1 > self.dropnet) | (num_faces.unsqueeze(1).eq(1))).to(
                    torch.float32
                )
                aggregated_feats = [
                    aggregated_feats[0] * mask_1,
                    aggregated_feats[1] * mask_2,
                ]
        aggregated_feat = torch.cat(aggregated_feats, dim=1)
        output = self.graph_pred_linear(aggregated_feat)
        return output

