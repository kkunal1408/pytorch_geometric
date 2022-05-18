import os
import torch
import json
import logging
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score
from torch_geometric.nn import (GCNConv,
GATConv)
from debug import debug, plot_results

import csv
import argparse

if not os.path.exists('LOG'):
    os.makedirs('LOG')
logging.basicConfig(filename='LOG/logfile.log', level=logging.DEBUG)
if os.path.exists('LOG/logfile.log'):
    os.remove('LOG/logfile.log')

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--num_layers', default=4, help="number of layers")
parser.add_argument('-epochs', '--epochs', default=20, help="number of epochs")
parser.add_argument('-hidden_channels', '--hidden_channels', default=16, help="number of channels in hidden layers")
args = parser.parse_args()

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'PPI_GANA')
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


class GanaGCN(torch.nn.Module):
    def __init__(self, num_layers, hidden_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(train_dataset.num_features, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, train_dataset.num_classes))
        print(f"GanaGCN: # of layers:{len(self.convs)}, hidden layers:{hidden_channels}")


    def forward(self, data):
        x = data.x
        for conv in self.convs:
            x1 = conv(x, data.edge_index)
            x = F.relu(x1)
            x = F.dropout(x, training=self.training)
        return x1


class GanaGCNEdgeWeight(torch.nn.Module):
    def __init__(self, num_layers, hidden_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(train_dataset.num_features, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, train_dataset.num_classes))
        print(
            f"GanaGCNEdgeWeight: # of layers:{len(self.convs)}, hidden layers:{hidden_channels}")

    def forward(self, data):
        x=data.x
        for conv in self.convs:
            x1 = conv(x, data.edge_index, data.edge_weight)
            x = F.relu(x1)
            x = F.dropout(x, training=self.training)
        return x1

class GanaGAT(torch.nn.Module):
    def __init__(self, num_layers, hidden_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GATConv(train_dataset.num_features, hidden_channels, heads=8, dropout=0.6))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * 8, hidden_channels * 8,
                        heads=1, concat=True, dropout=0.6))
        self.convs.append(GATConv(hidden_channels * 8, train_dataset.num_classes,
                                  heads=1, concat=False, dropout=0.6))

        print(
            f"GanaGAT: # layers k:{len(self.convs)}, hidden layers :{hidden_channels}")

    def forward(self, data):
        x = data.x
        for conv in self.convs:
            x1 = conv(x, data.edge_index)
            x = F.elu(x1)
            x = F.dropout(x, p=0.6, training=self.training)
        return x1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_op = torch.nn.BCEWithLogitsLoss()
summary_path = f'results/summary.json'
if os.path.exists(summary_path):
    with open(summary_path, 'r') as f:
        summary = json.load(f)
else:
    summary = {}

regression_models = [
    GanaGCN,
    # GanaGAT,
    GanaGCNEdgeWeight
]
for Net in regression_models:
    model = Net(int(args.num_layers), int(args.hidden_channels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    def train():
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = loss_op(model(data), data.y)
            total_loss += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        return total_loss / len(train_loader.dataset)

    @torch.no_grad()
    def test(loader):
        model.eval()

        ys, preds = [], []
        for data in loader:
            ys.append(data.y)
            data = data.to(device)
            out = model(data)
            preds.append((out > 0).float().cpu())

        y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
        return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

    epoch_list = []
    loss_list = []
    val_list = []
    test_list = []
    dict_key = '_'.join(
        [Net.__name__, str(args.num_layers), str(args.epochs), str(args.hidden_channels)])
    # if dict_key in summary.keys():
    #     logging.info(f"skipping run {dict_key} as summary exists")
    #     continue
    for epoch in range(1, int(args.epochs)):
        loss = train()
        val_f1 = test(val_loader)
        test_f1 = test(test_loader)
        print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
            epoch, loss, val_f1, test_f1))
        logging.debug('Epoch: {: 02d}, Loss: {: .4f}, Val: {: .4f}, Test: {: .4f}'.format(
            epoch, loss, val_f1, test_f1))
        epoch_list.append(int(epoch))
        loss_list.append(float(loss))
        val_list.append(float(val_f1))
        test_list.append(float(test_f1))


    summary[dict_key] = {'epoch': epoch_list,
                        'loss': loss_list, 'val': val_list, 'test': test_list}
    result_dir = f"results/{Net.__name__}/{args.num_layers}_{args.hidden_channels}"
    os.makedirs(result_dir, exist_ok=True)
    debug(val_loader, result_dir, model)
    plot_results(1, int(args.epochs) - 1, 'test accuracy', 'test',
                 f'results/{Net.__name__}_test_comparison.png',summary)
    plot_results(1, int(args.epochs) - 1, 'train loss', 'loss',
                 f'results/{Net.__name__}_train_loss_comparison.png', summary)
    plot_results(1, int(args.epochs) - 1, 'validation accuracy', 'val',
                 f'results/{Net.__name__}_val_comparison.png', summary)

if summary:
	if os.path.exists(summary_path):
		os.remove(summary_path)
		with open(summary_path, 'w') as f:
			json.dump(summary, f, indent=4)
