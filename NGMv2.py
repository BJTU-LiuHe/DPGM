import torch.nn
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import SplineConv
from torch_geometric.data import Data

class SConv(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super(SConv, self).__init__()

        self.in_channels = input_features
        self.num_layers = 2
        self.convs = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            conv = SplineConv(input_features, output_features, dim=2, kernel_size=5, aggr="max")
            self.convs.append(conv)
            input_features = output_features

        input_features = output_features
        self.out_channels = input_features
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        xs = [x]

        for conv in self.convs[:-1]:
            xs += [F.relu(conv(xs[-1], edge_index, edge_attr))]

        xs += [self.convs[-1](xs[-1], edge_index, edge_attr)]
        return xs[-1]


class SiameseSConvOnNodes(torch.nn.Module):
    def __init__(self, input_node_dim):
        super(SiameseSConvOnNodes, self).__init__()
        self.num_node_features = input_node_dim
        self.mp_network = SConv(input_features=self.num_node_features, output_features=self.num_node_features)

    def forward(self, graph):
        old_features = graph.x
        result = self.mp_network(graph)
        graph.x = old_features + 0.1 * result
        return graph


class SiameseNodeFeaturesToEdgeFeatures(torch.nn.Module):
    def __init__(self, total_num_nodes):
        super(SiameseNodeFeaturesToEdgeFeatures, self).__init__()
        self.num_edge_features = total_num_nodes

    def forward(self, graph, hyperedge=False):
        orig_graphs = self.vertex_attr_to_edge_attr(graph)
        return orig_graphs

    def vertex_attr_to_edge_attr(self, graph):
        """Assigns the difference of node features to each edge"""
        flat_edges = graph.edge_index.transpose(0, 1).reshape(-1)
        vertex_attrs = torch.index_select(graph.x, dim=0, index=flat_edges)

        new_shape = (graph.edge_index.shape[1], 2, vertex_attrs.shape[1])
        vertex_attrs_reshaped = vertex_attrs.reshape(new_shape).transpose(0, 1)
        new_edge_attrs = vertex_attrs_reshaped[0] - vertex_attrs_reshaped[1]
        graph.edge_attr = new_edge_attrs
        return graph

    def vertex_attr_to_hyperedge_attr(self, graph):
        """Assigns the angle of node features to each hyperedge.
           graph.hyperedge_index is the incidence matrix."""
        flat_edges = graph.hyperedge_index.transpose(0, 1).reshape(-1)
        vertex_attrs = torch.index_select(graph.x, dim=0, index=flat_edges)

        new_shape = (graph.hyperedge_index.shape[1], 3, vertex_attrs.shape[1])

        vertex_attrs_reshaped = vertex_attrs.reshape(new_shape).transpose(0, 1)
        v01 = vertex_attrs_reshaped[0] - vertex_attrs_reshaped[1]
        v02 = vertex_attrs_reshaped[0] - vertex_attrs_reshaped[2]
        v12 = vertex_attrs_reshaped[1] - vertex_attrs_reshaped[2]
        nv01 = torch.norm(v01, p=2, dim=-1)
        nv02 = torch.norm(v02, p=2, dim=-1)
        nv12 = torch.norm(v12, p=2, dim=-1)

        cos1 = torch.sum(v01 * v02, dim=-1) / (nv01 * nv02)
        cos2 = torch.sum(-v01 * v12, dim=-1) / (nv01 * nv12)
        cos3 = torch.sum(-v12 * -v02, dim=-1) / (nv12 * nv02)

        graph.hyperedge_attr = torch.stack((cos1, cos2, cos3), dim=-1)
        return graph


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=1024)
        # self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
        #     total_num_nodes=self.message_pass_node_features.num_node_features
        # )

    def to_pyg_graph(self, A, Pts, desc):
        rescale = 256

        edge_feat = 0.5 * (
                    np.expand_dims(Pts, axis=1) - np.expand_dims(Pts, axis=0)) / rescale + 0.5  # from Rolink's paper
        edge_index = np.nonzero(A)
        edge_attr = edge_feat[edge_index]

        edge_attr = np.clip(edge_attr, 0, 1)
        assert (edge_attr > -1e-5).all(), Pts

        if torch.cuda.is_available():
            pyg_graph = Data(
                x=torch.tensor(desc).to(torch.float32).cuda(),
                edge_index=torch.tensor(np.array(edge_index), dtype=torch.long).cuda(),
                edge_attr=torch.tensor(edge_attr).to(torch.float32).cuda(),
            )
        else:
            pyg_graph = Data(
                x=torch.tensor(desc).to(torch.float32),
                edge_index=torch.tensor(np.array(edge_index), dtype=torch.long),
                edge_attr=torch.tensor(edge_attr).to(torch.float32),
            )
        return pyg_graph

    def forward(
        self,
        descs, tails, heads, pts,
    ):
        n_node = pts.shape[0]
        n_edge = len(tails)
        A = np.zeros((n_node, n_node))
        for i in range(n_edge):
            A[tails[i], heads[i]] = 1.0

        graph = self.to_pyg_graph(A, pts, descs)
        graph = self.message_pass_node_features(graph)

        return graph.x.data.cpu().numpy()
