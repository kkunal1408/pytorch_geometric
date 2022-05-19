
import os
import json
import matplotlib.pylab as plt
import time
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix
from torch_geometric.utils import to_networkx
import networkx as nx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def debug(loader, result_dir, model):
    model.eval()
    assert os.path.exists(
        '../data/PPI_GANA/raw/valid_name_map.json'), f'no map file found '
    with open('../data/PPI_GANA/raw/valid_name_map.json', "r") as f:
        map_data = json.loads(json.load(f))
    x_names = [map_data["name"][str(i)] for i in range(len(map_data['name']))]
    count = 0


    for data in loader:
        plt.figure(figsize=(12, 12))
        start = time.time()
        out = model(data.to(device))
        end = time.time()
        eles, x_names = x_names[:len(data.y)], x_names[len(data.y):]
        # y_val = [row.index(1) for row in data.y.tolist()]
        # y_pred = [row.index(1) for row in (out > 0).tolist()]
        if len(data.y.size())> 1:
            y_val = np.argmax(data.y, axis=1).to(torch.long).tolist()
        else:
            y_val = data.y
        y_pred = out.argmax(dim=-1)
        df = pd.DataFrame({'actual': y_val, 'predicted': y_pred})
        df['true'] = (df['actual'] != df['predicted'])
        # print(df)
        cm = confusion_matrix(y_val, y_pred)
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        f1 = f1_score(y_val, y_pred, average='micro')
        G = to_networkx(data, node_attrs=['y'], to_undirected=True)
        mapping = {k: i for k, i in enumerate(eles)}
        incorrect_nodes = [v for k, v in mapping.items() if df['true'][k]]
        G = nx.relabel_nodes(G, mapping)
        node_kwargs = {}
        node_kwargs['node_size'] = 800
        node_kwargs['cmap'] = 'cool'
        label_kwargs = {}
        label_kwargs['font_size'] = 10
        ax = plt.gca()
        pos = nx.spring_layout(G, seed=1)
        for source, target, _ in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="-",
                ))

        nx.draw_networkx_nodes(G, pos, node_color=y_pred)
        nx.draw_networkx_nodes(
            G, pos, nodelist=incorrect_nodes, node_color="tab:red")
        nx.draw_networkx_labels(G, pos, **label_kwargs)
        df['name'] = eles
        plt_title = f"f1 score: {format(f1,'.2')} TPR: {TPR} FPR: {FPR} time: {format(1000*(end - start), '.4')}"
        plt.title(plt_title)
        print(plt_title)
        plt.savefig(
            f"{result_dir}/{str(count)}.png")
        df.to_csv(
            f"{result_dir}/{str(count)}.csv")
        count += 1
        # if count %20 == 5:
        break


def plot_results(xlimit, ylimit, title, dict_key, fpath, summary):
	fig, ax = plt.subplots()
	for key, value in summary.items():
		ax.plot(value[dict_key], label=key)
	ax.set_ylim(0, xlimit)
	ax.set_xlim(0, ylimit)
	ax.set_ylabel(title, fontsize=14, fontweight='bold')
	ax.set_xlabel('# Epochs', fontsize=14, fontweight='bold')
	ax.set_title(f'{title} with epochs', fontsize=14, fontweight='bold')
	ax.legend()
	if os.path.exists(fpath):
		os.remove(fpath)
	fig.savefig(fpath)
