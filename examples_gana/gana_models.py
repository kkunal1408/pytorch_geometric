from torch_geometric.utils import normalized_cut, degree
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (GCNConv,
GATConv,
ChebConv,
GCN2Conv,
SAGEConv,
RGATConv,
graclus,
max_pool,
max_pool_x,
global_mean_pool,
RGCNConv)


class GanaGCN(torch.nn.Module):
    def __init__(self, num_layers, hidden_channels, num_features, num_classes):
        super().__init__()
        torch.manual_seed(12345)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, num_classes))
        print(f"GanaGCN: # of layers:{len(self.convs)}, hidden layers:{hidden_channels}")

    def forward(self, data):
        x = data.x
        for conv in self.convs:
            x1 = conv(x, data.edge_index)
            x = F.relu(x1)
            x = F.dropout(x, training=self.training)
        return F.log_softmax(x1, dim=1)


class GanaGCN2(torch.nn.Module):
    def __init__(self, num_layers, hidden_channels, num_features, num_classes, alpha=0.5, theta=1.0, shared_weights=True, dropout=0.0):
        super().__init__()
        torch.manual_seed(12345)
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, num_classes))

        self.convs = torch.nn.ModuleList()

        for layer in range(num_layers):
            self.convs.append(GCN2Conv(hidden_channels, alpha, theta, layer+1, shared_weights, normalize=False))
        self.dropout = dropout
        print(
            f"GanaGCN2: # of layers:{len(self.convs)}, hidden layers:{hidden_channels}")

    def forward(self, data):
        x = F.dropout(data.x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        for conv in self.convs:
            h = F.dropout(x, self.dropout, training=self.training)
            h = conv(h, x_0, data.adj_t)
            x = h + x
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        return F.log_softmax(x, dim=1)

class GanaGCNEdgeWeight(torch.nn.Module):
    def __init__(self, num_layers, hidden_channels, num_features, num_classes):
        super().__init__()
        torch.manual_seed(12345)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, num_classes))
        print(
            f"GanaGCNEdgeWeight: # of layers:{len(self.convs)}, hidden layers:{hidden_channels}")

    def forward(self, data):
        x=data.x
        for conv in self.convs:
            x1 = conv(x, data.edge_index, data.edge_weight)
            x = F.relu(x1)
            x = F.dropout(x, training=self.training)
        return F.log_softmax(x1, dim=1)

class GanaGAT(torch.nn.Module):
    def __init__(self, num_layers, hidden_channels, num_features, num_classes):
        super().__init__()
        torch.manual_seed(12345)
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GATConv(num_features, hidden_channels, heads=8, dropout=0.6))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * 8, hidden_channels * 8,
                        heads=1, concat=True, dropout=0.6))
        self.convs.append(GATConv(hidden_channels * 8, num_classes,
                                  heads=1, concat=False, dropout=0.6))

        print(
            f"GanaGAT: # layers k:{len(self.convs)}, hidden layers :{hidden_channels}")

    def forward(self, data):
        x = data.x
        for conv in self.convs:
            x1 = conv(x, data.edge_index)
            x = F.elu(x1)
            x = F.dropout(x, p=0.6, training=self.training)
        return F.log_softmax(x1, dim=1)


class GanaChebConv(torch.nn.Module):
    def __init__(self, num_layers, hidden_channels, num_features, num_classes):
        super().__init__()
        torch.manual_seed(12345)
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            ChebConv(num_features, hidden_channels, K=4))
        for _ in range(num_layers - 2):
            self.convs.append(
                ChebConv(hidden_channels, hidden_channels, K=4))
        self.convs.append(
            ChebConv(hidden_channels, num_classes, K=4))

        print(
            f"GanaGAT: # layers k:{len(self.convs)}, hidden layers :{hidden_channels}")

    def forward(self, data):
        x = data.x
        for conv in self.convs:
            x1 = conv(x, data.edge_index)
            x = F.relu(x1)
            x = F.dropout(x, p=0.6, training=self.training)
        return F.log_softmax(x1, dim=1)


class GanaSAGEConv(torch.nn.Module):
    def __init__(self, num_layers, hidden_channels, num_features, num_classes):
        super().__init__()
        torch.manual_seed(12345)
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            SAGEConv(num_features, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, num_classes))

        print(
            f"GanaGAT: # layers k:{len(self.convs)}, hidden layers :{hidden_channels}")

    def forward(self, data):
        x = data.x
        for conv in self.convs:
            x1 = conv(x, data.edge_index)
            x = F.relu(x1)
            x = F.dropout(x, p=0.6, training=self.training)
        return F.log_softmax(x1, dim=1)


class GanaRGATConv(torch.nn.Module):
    def __init__(self, num_layers, hidden_channels, num_features, num_classes=2,num_relations=6):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            RGATConv(num_features, hidden_channels,
                     num_relations))
        for _ in range(num_layers - 2):
            self.convs.append(
                RGATConv(hidden_channels, hidden_channels, num_relations))
        self.convs.append(
            RGATConv(hidden_channels, num_classes, num_relations))
        print(
            f"number of layers k:{len(self.convs)} hidden layersize :{hidden_channels}")

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        for conv in self.convs:
            x1 = conv(x, edge_index, edge_type)

            x = F.relu(x1)
            x = F.dropout(x, training=self.training)

        return F.log_softmax(x1, dim=1)


class GanaRGCNConv(torch.nn.Module):
    def __init__(self, num_layers, hidden_channels, num_features, num_classes=2, num_relations=6):
        super().__init__()
        torch.manual_seed(12345)
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            RGCNConv(num_features, hidden_channels,
                     num_relations, num_bases=30))
        for _ in range(num_layers - 2):
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=30))
        self.convs.append(
            RGCNConv(hidden_channels, num_classes, num_relations, num_bases=30))
        print(
            f"number of layers k:{len(self.convs)} hidden layersize :{hidden_channels}")

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        for conv in self.convs:
            x1 = conv(x, edge_index, edge_type)

            x = F.relu(x1)
            x = F.dropout(x, training=self.training)

        return F.log_softmax(x1, dim=1)


class GanaOrg(torch.nn.Module):
    def __init__(self, num_layers, hidden_channels, num_features, num_classes):
        super().__init__()
        torch.manual_seed(12345)
        hops = num_layers
        self.conv1 = ChebConv(num_features, hidden_channels, K=hops)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=hops)
        # self.fc1 = torch.nn.Linear(hidden_channels, 128)
        # self.fc2 = torch.nn.Linear(128, num_classes)
        print(
            f"Gana_original: # layers k:2, hidden layers :{hidden_channels}")

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_weight))
        # eweight = data.edge_weight
        # print(data.edge_index)
        weight = normalized_cut(data.edge_index, data.edge_weight)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        # data.edge_weight = degree(data.edge_index)
        data, map = max_pool(cluster, data)
        # node_degree = degree(data.edge_index[1],
        #                           num_nodes=data.batch.size(0), dtype=torch.long)
        # data.edge_weight = torch.tensor(
        #     [node_degree[_edge] for _edge in data.edge_index[1]],
        #     dtype=torch.float)
        data.edge_weight = torch.ones(len(data.edge_index[0]))
        # new_weight = torch.zeros(len(data.edge_index[0],0), dtype=torch.float)
        # for i in range(len(new_weight)):
        #     new_weight[i] = sum(for sp in data.edge_index[0])
        # print(eweight, map, data.edge_index)

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_weight))
        weight = normalized_cut(data.edge_index, data.edge_weight)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data, map1 = max_pool(cluster, data)
        remap1 = [int(map1[v]) for k, v in enumerate(map)]

        # x, batch = max_pool_x(cluster, data.x, data.batch)
        # x = global_mean_pool(x, batch)
        # x = F.elu(self.fc1(x))
        remap = torch.stack([data.x[v] for k, v in enumerate(remap1)])
        return F.log_softmax(remap, dim=1)

