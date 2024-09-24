# https://github.com/divelab/DIG/blob/dig/dig/xgraph/dataset/syn_dataset.py

import os
import yaml
import torch
import pickle
import numpy as np
import os.path as osp
from pathlib import Path
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, download_url
from .syn_utils.gengraph import *
from torch_geometric.utils import from_networkx
from sklearn.model_selection import train_test_split

from utils.cell_utils import convert_graph_dataset_with_rings
from utils.complexdataset import InMemoryComplexDataset 

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


class BaHouseGrid(InMemoryComplexDataset):
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
        "ba_house_grid": ["BA_House_Grid", "BA_House_Grid.pkl", "BA_House_Grid"],
        "ba_house_and_grid": ["BA_House_And_Grid", "BA_House_And_Grid.pkl", "BA_House_And_Grid"],
        "ba_house_or_grid": ["BA_House_Or_Grid", "BA_House_Or_Grid.pkl", "BA_House_Or_Grid"],
    }

    def __init__(
        self, root, name, max_ring_size, transform=None, pre_transform=None,include_down_adj=True, init_method='sum', n_jobs=1,
        num_graphs=2000, num_shapes=1, width_basis=20, nnf=1,
        seed=2, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
    ):
        self.name = name.lower()
        self._max_ring_size = max_ring_size
        self._use_edge_features = True
        self._n_jobs = n_jobs
        self.num_graphs = num_graphs
        self.num_shapes = num_shapes
        self.width_basis = width_basis
        self.seed = seed
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.train_ratio = train_ratio
        self.nnf = nnf # num node features
        super(BaHouseGrid, self).__init__(root, transform, pre_transform, max_dim=2,
                                              init_method=init_method, include_down_adj=include_down_adj,
                                              cellular=True)
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
    
    @property
    def processed_dir(self):
        """This is overwritten, so the cellular complex data is placed in another folder"""
        prefix = "cell_" if self._cellular else ""
        directory = osp.join(self.root, self.name, f'{prefix}complex_dim{self.max_dim}_{self._init_method}')
        suffix1 = f"_{self._max_ring_size}rings" if self._cellular else ""
        suffix2 = "-E" if self._use_edge_features else ""
        return directory + suffix1 + suffix2

    def download(self):
         if self.name.lower() == "BA_2Motifs".lower():
            url = self.url.format(self.names[self.name][1])
            download_url(url, self.raw_dir)
    
    def load_dataset(self):
        print("Loading dataset from disk...")
        data, slices = torch.load(self.processed_paths[0])
        idx = torch.load(self.processed_paths[1])
        return data, slices, idx

    def process(self):
        
        if self.name.lower() == "BA_House_Grid".lower():
            motifs = ["_house", "_grid"]
            labels = [0, 1]
            probs = [0.5, 0.5]

            data_list = []

            for graph_idx in range(self.num_graphs):
                idx = np.random.choice(list(range(len(motifs))), p=probs)
                name = motifs[idx]
                generate_function = "gen_ba" + name
                # print(idx, name, generate_function)
                # nb_shapes = np.random.randint(1, self.num_shapes)
                G, role_idx, _ = eval(generate_function)(
                    nb_shapes=1,
                    width_basis=self.width_basis,
                    m=1,
                    feature_generator=featgen.ConstFeatureGen(
                        np.ones(self.nnf, dtype=float)
                    ), 
                    is_weighted=True,
                )
                data = self.from_G_to_data(G, graph_idx, labels[idx], name)
                node_label = torch.tensor(role_idx, dtype=torch.float)
                node_label[node_label != 0] = 1
                data.node_label = node_label
                data_list.append(data)
        
        elif self.name.lower() in ["ba_house_and_grid", "ba_house_or_grid"]:
            # Binary graph classification task
            motifs = ["", "_house", "_grid", "_house_grid"]
            if "and" in self.name.lower():
                labels = [0, 0, 0, 1]
                probs = [0.5/3, 0.5/3, 0.5/3, 0.5]
            elif "or" in self.name.lower():
                labels = [0, 1, 1, 1]
                probs = [0.5, 0.5/3, 0.5/3, 0.5/3]

            data_list = []
            for graph_idx in range(self.num_graphs):
                idx = np.random.choice(list(range(len(motifs))), p=probs)
                name = motifs[idx]
                generate_function = "gen_ba" + name
                # print(idx, name, generate_function)
                # nb_shapes = np.random.randint(1, self.num_shapes)
                m = np.random.randint(1, 3)
                G, role_idx, _ = eval(generate_function)(
                    nb_shapes=1,
                    width_basis=self.width_basis,
                    m=m,
                    feature_generator=featgen.ConstFeatureGen(
                        np.ones(self.nnf, dtype=float)
                    ), 
                    is_weighted=True,
                )
                data = self.from_G_to_data(G, graph_idx, labels[idx], name)
                node_label = torch.tensor(role_idx, dtype=torch.float)
                node_label[node_label != 0] = 1
                data.node_label = node_label
                data_list.append(data)

        else:
            generate_function = "gen_" + self.name

            G, labels, name = eval(generate_function)(
                nb_shapes=self.num_shapes,
                width_basis=self.width_basis,
                feature_generator=featgen.ConstFeatureGen(
                    np.ones(self.nnf, dtype=float)
                ),
            )

            data = from_networkx(G.to_undirected(), all)
            data.adj = torch.LongTensor(nx.to_numpy_matrix(G))
            data.num_classes = len(np.unique(labels))
            data.y = torch.LongTensor(labels)
            data.x = data.x.float()
            data.edge_attr = torch.ones(data.edge_index.size(1))
            n = data.num_nodes
            data.train_mask, data.val_mask, data.test_mask = (
                torch.zeros(n, dtype=torch.bool),
                torch.zeros(n, dtype=torch.bool),
                torch.zeros(n, dtype=torch.bool),
            )
            train_ids, test_ids = train_test_split(
                range(n),
                test_size=self.test_ratio,
                random_state=self.seed,
                shuffle=True,
            )
            train_ids, val_ids = train_test_split(
                train_ids,
                test_size=self.val_ratio,
                random_state=self.seed,
                shuffle=True,
            )

            data.train_mask[train_ids] = 1
            data.val_mask[val_ids] = 1
            data.test_mask[test_ids] = 1

            data = data if self.pre_transform is None else self.pre_transform(data)
            data_list = [data]

        idx = get_random_split_idx(dataset=data_list)

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

        # torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.names[self.name][0], len(self))
    
    def from_G_to_data(self, G, graph_idx, label, name='_house'):
        # attr_list = [str(attr) for attr in list(nx.get_edge_attributes(G, 'weight').values())]
        attr_list = nx.get_edge_attributes(G, 'weight').values()
        data = from_networkx(G, group_edge_attrs=all)
        data.x = data.feat.float()
        # adj = torch.LongTensor(nx.to_numpy_matrix(G))
        data.y = torch.tensor(label).float().reshape(-1, 1)
        data.edge_label = torch.squeeze(data.edge_attr)
        data.edge_attr = torch.ones(data.edge_index.size(1), 1)
        data.idx = graph_idx
        return data

