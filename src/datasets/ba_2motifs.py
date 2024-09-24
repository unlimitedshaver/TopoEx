#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.cell_utils import convert_graph_dataset_with_rings
from utils.complexdataset import InMemoryComplexDataset 

import os.path as osp
import os
import pickle

import numpy as np

from torch_geometric.utils import dense_to_sparse

import pandas as pd
import torch
# from ogb.utils.url import decide_download
from torch_geometric.data import Data
from utils import  download_url



def get_random_split_idx(dataset, splits={'train':0.8, 'valid':0.1, 'test':0.1
                                          }, random_state=0, mutag_x=False):
    if random_state is not None:
        np.random.seed(random_state)

    print('[INFO] Split dataset!')
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)

    if not mutag_x:
        n_train, n_valid = int(splits['train'] * len(idx)), int(splits['valid'] * len(idx))
        train_idx = idx[:n_train]
        valid_idx = idx[n_train:n_train+n_valid]
        test_idx = idx[n_train+n_valid:]
    else:
        print('[INFO] mutag_x is True!')
        n_train = int(splits['train'] * len(idx))
        train_idx, valid_idx = idx[:n_train], idx[n_train:]
        test_idx = [i for i in range(len(dataset)) if (dataset[i].y.squeeze() == 0 and dataset[i].edge_label.sum() > 0)]
    return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}


def read_ba2motif_data(folder: str, prefix):
    with open(os.path.join(folder, f"{prefix}.pkl"), 'rb') as f:
        dense_edges, node_features, graph_labels = pickle.load(f)

    data_list = []
    for graph_idx in range(dense_edges.shape[0]):
        x = torch.from_numpy(node_features[graph_idx]).float()
        edge_index = dense_to_sparse(torch.from_numpy(dense_edges[graph_idx]))[0]
        # edge_index = torch.from_numpy(dense_edges[graph_idx])
        y = torch.from_numpy(np.where(graph_labels[graph_idx])[0]).reshape(-1, 1).float()

        node_label = torch.zeros(x.shape[0]).float()
        node_label[20:] = 1
        edge_label = ((edge_index[0] >= 20) & (edge_index[0] < 25) & (edge_index[1] >= 20) & (edge_index[1] < 25)).float() 
        edge_label = torch.tensor(edge_label) ##

        data_list.append(Data(x=x, edge_index=edge_index, y=y, node_label=node_label, edge_label=edge_label))
    return data_list


class SynGraphDataset(InMemoryComplexDataset):
    r"""
    The Synthetic datasets used in
    `Parameterized Explainer for Graph Neural Network <https://arxiv.org/abs/2011.04573>`_.
    It takes Barabási–Albert(BA) graph or balance tree as base graph
    and randomly attachs specific motifs to the base graph.
    Args:
        root (:obj:`str`): Root data directory to save datasets
        name (:obj:`str`): The name of the dataset. Including :obj:`BA_shapes`, BA_grid,
        transform (:obj:`Callable`, :obj:`None`): A function/transform that takes in an
            :class:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (:obj:`Callable`, :obj:`None`):  A function/transform that takes in
            an :class:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://github.com/divelab/DIG_storage/raw/main/xgraph/datasets/{}'
    # Format: name: [display_name, url_name, filename]
    names = {
        'ba_shapes': ['BA_shapes', 'BA_shapes.pkl', 'BA_shapes'],
        'ba_community': ['BA_Community', 'BA_Community.pkl', 'BA_Community'],
        'tree_grid': ['Tree_Grid', 'Tree_Grid.pkl', 'Tree_Grid'],
        'tree_cycle': ['Tree_Cycle', 'Tree_Cycles.pkl', 'Tree_Cycles'],
        'ba_2motifs': ['BA_2Motifs', 'BA_2Motifs.pkl', 'BA_2Motifs']
    }

    def __init__(self, root, name, max_ring_size, transform=None, pre_transform=None,
                 include_down_adj=True, init_method='sum', n_jobs=32 ):
        
        self._max_ring_size = max_ring_size
        self._use_edge_features = True
        self._n_jobs = n_jobs
        self.name = name.lower()
        super(SynGraphDataset, self).__init__(root, transform, pre_transform, max_dim=2,
                                              init_method=init_method, include_down_adj=include_down_adj,
                                              cellular=True)
        # self.data, self.slices = torch.load(self.processed_paths[0])
        self.data, self.slices, idx = self.load_dataset()
        self.train_ids = idx['train']
        self.val_ids = idx['valid']
        self.test_ids = idx['test']

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return f'{self.names[self.name][2]}.pkl'

    @property
    def processed_file_names(self):
        return [f'{self.name}_complex.pt', f'{self.name}_idx.pt']
        # return ['data.pt']

    @property
    def processed_dir(self):
        """This is overwritten, so the cellular complex data is placed in another folder"""
        prefix = "cell_" if self._cellular else ""
        directory = osp.join(self.root, self.name, f'{prefix}complex_dim{self.max_dim}_{self._init_method}')
        suffix1 = f"_{self._max_ring_size}rings" if self._cellular else ""
        suffix2 = "-E" if self._use_edge_features else ""
        return directory + suffix1 + suffix2
        

    def download(self):
        url = self.url.format(self.names[self.name][1])
        download_url(url, self.raw_dir)
    
    def load_dataset(self):
        print("Loading dataset from disk...")
        data, slices = torch.load(self.processed_paths[0])
        idx = torch.load(self.processed_paths[1])
        return data, slices, idx

    def process(self):

        print(f"Processing cell complex dataset for {self.name}")
        if self.name.lower() == 'BA_2Motifs'.lower():
            data_list = read_ba2motif_data(self.raw_dir, self.names[self.name][2])
            idx = get_random_split_idx(dataset=data_list)

        #     if self.pre_filter is not None:
        #         data_list = [self.get(idx) for idx in range(len(self))]
        #         data_list = [data for data in data_list if self.pre_filter(data)]
        #         self.data, self.slices = self.collate(data_list)

        #     if self.pre_transform is not None:
        #         data_list = [self.get(idx) for idx in range(len(self))]
        #         data_list = [self.pre_transform(data) for data in data_list]
        #         self.data, self.slices = self.collate(data_list)
        # else:
        #     # Read data into huge `Data` list.
        #     data = self.read_syn_data()
        #     data = data if self.pre_transform is None else self.pre_transform(data)
        #     data_list = [data]

        print(f"Converting the {self.name} dataset to a cell complex...") 
        complexes, _, _ = convert_graph_dataset_with_rings(
            data_list,
            max_ring_size=self._max_ring_size,
            include_down_adj=self.include_down_adj,
            init_method=self._init_method,
            init_edges=self._use_edge_features,
            init_rings=False,
            n_jobs=self._n_jobs,
            v_label=True, e_label=True)

        print(f'Saving processed dataset in {self.processed_paths[0]}...')
        torch.save(self.collate(complexes, 2), self.processed_paths[0])
        
        print(f'Saving idx in {self.processed_paths[1]}...')
        torch.save(idx, self.processed_paths[1])

    def __repr__(self):
        return '{}({})'.format(self.names[self.name][0], len(self))

    def read_syn_data(self):
        with open(self.raw_paths[0], 'rb') as f:
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pickle.load(f)

        x = torch.from_numpy(features).float()
        y = train_mask.reshape(-1, 1) * y_train + val_mask.reshape(-1, 1) * y_val + test_mask.reshape(-1, 1) * y_test
        y = torch.from_numpy(np.where(y)[1])
        edge_index = dense_to_sparse(torch.from_numpy(adj))[0]
        data = Data(x=x, y=y, edge_index=edge_index)
        data.train_mask = torch.from_numpy(train_mask)
        data.val_mask = torch.from_numpy(val_mask)
        data.test_mask = torch.from_numpy(test_mask)
        return data

# if __name__ == '__main__':
    # data_config = yaml.safe_load(open('../configs/ba_2motifs.yml'))['data']
    # dataset = PeptidesFunctionalDataset(root='../../data/ba_2motifs', max_ring_size=data_config['max_ring_size'])