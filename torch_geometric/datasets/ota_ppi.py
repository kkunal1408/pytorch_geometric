from itertools import product
import os
import os.path as osp
import json

import torch
import numpy as np
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)
from torch_geometric.utils import remove_self_loops


class OTA_PPI(InMemoryDataset):
    r"""The protein-protein interaction networks from the `"Predicting
    Multicellular Function through Multi-layer Tissue Networks"
    <https://arxiv.org/abs/1707.04638>`_ paper, containing positional gene
    sets, motif gene sets and immunological signatures as features (50 in
    total) and gene ontology sets as labels (121 in total).

    Args:
        root (string): Root directory where the dataset should be saved.
        split (string): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    # url = 'https://data.dgl.ai/dataset/ppi.zip'
    url = 'https://github.com/kkunal1408/check/blob/master/'

    def __init__(self, root, split='train', transform=None, pre_transform=None,
                 pre_filter=None):

        assert split in ['train', 'val', 'test']

        super(OTA_PPI, self).__init__(
            root, transform, pre_transform, pre_filter)

        if split == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        # elif split == 'val':
        #     self.data, self.slices = torch.load(self.processed_paths[1])
        # elif split == 'test':
        #     self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        # splits = ['train', 'valid', 'test']
        # files = ['feats.npy', 'graph_id.npy', 'graph.json', 'labels.npy']
        # return ['{}_{}'.format(s, f) for s, f in product(splits, files)]
        return ["processed_data.p"]

    @property
    def processed_file_names(self):
        # return ['train.pt', 'val.pt', 'test.pt']
        return 'data.pt'

    def download(self):
        # path = download_url(self.url, self.root)
        # extract_zip(path, self.raw_dir)
        # os.unlink(path)
        for name in self.raw_file_names:
            download_url('{}/{}'.format(self.url, name), self.raw_dir)

    def process(self):
        import networkx as nx
        import pickle
        from torch_geometric.utils import from_scipy_sparse_matrix

        # from networkx.readwrite import json_graph
        path = osp.join(self.raw_dir, 'processed_data.p')
        print(path)
        with open(path, 'rb') as f:
            all_designs = pickle.load(f)
        data_list = []
        for circuit_name, circuit_data in all_designs.items():
            df = circuit_data["data_matrix"]
            # print(circuit_name)
            node_features = df.values
            # print(node_features, type(node_features))
            node_features = np.delete(node_features, 0, 1)
            node_features = np.array(node_features, dtype=np.int8)
            all_x = node_features[:, 0:16]
            all_y = circuit_data["target"].astype(np.int_)
            y_onehot = np.zeros((all_y.size, all_y.max()+1))
            y_onehot[np.arange(all_y.size),all_y]=1
            if len(y_onehot.shape)==1:
                continue

            adj = circuit_data["adjacency_matrix"]
            edge_index, edge_weight = from_scipy_sparse_matrix(adj)
            print(torch.tensor(all_y.astype(np.int_)))
            data = Data(edge_index=edge_index, x=torch.Tensor(
                all_x), y=torch.tensor(all_y))
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])
        # for s, split in enumerate(['train', 'valid', 'test']):
        #     path = osp.join(self.raw_dir, '{}_graph.json').format(split)
        #     with open(path, 'r') as f:
        #         G = nx.DiGraph(json_graph.node_link_graph(json.load(f)))

        #     x = np.load(osp.join(self.raw_dir, '{}_feats.npy').format(split))
        #     x = torch.from_numpy(x).to(torch.float)

        #     y = np.load(osp.join(self.raw_dir, '{}_labels.npy').format(split))
        #     y = torch.from_numpy(y).to(torch.float)

        #     data_list = []
        # path = osp.join(self.raw_dir, '{}_graph_id.npy').format(split)
        # idx = torch.from_numpy(np.load(path)).to(torch.long)
        # idx = idx - idx.min()

        # for i in range(idx.max().item() + 1):
        #     mask = idx == i

        #     G_s = G.subgraph(
        #         mask.nonzero(as_tuple=False).view(-1).tolist())
        #     edge_index = torch.tensor(list(G_s.edges)).t().contiguous()
        #     edge_index = edge_index - edge_index.min()
        #     edge_index, _ = remove_self_loops(edge_index)

        #     data = Data(edge_index=edge_index, x=x[mask], y=y[mask])

        #     if self.pre_filter is not None and not self.pre_filter(data):
        #         continue

        #     if self.pre_transform is not None:
        #         data = self.pre_transform(data)

        #     data_list.append(data)
        # torch.save(self.collate(data_list), self.processed_paths[s])
