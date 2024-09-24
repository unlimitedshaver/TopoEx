import os
import torch
import numpy as np
import os.path as osp
import pandas as pd
import pickle as pkl
from pathlib import Path
from torch_geometric.data import Data, download_url

from utils.cell_utils import convert_graph_dataset_with_rings
from utils.complexdataset import InMemoryComplexDataset


ATOM_TYPES = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "Na", "Ca", "I", "B", "H", "*"]


def edge_mask_from_node_mask(node_mask: torch.Tensor, edge_index: torch.Tensor):
    """
    Convert edge_mask to node_mask

    Args:
        node_mask (torch.Tensor): Boolean mask over all nodes included in edge_index. Indices must
            match to those in edge index. This is straightforward for graph-level prediction, but
            converting over subgraphs must be done carefully to match indices in both edge_index and
            the node_mask.
    """

    node_numbers = node_mask.nonzero(as_tuple=True)[0]

    iter_mask = torch.zeros((edge_index.shape[1],))

    # See if edges have both ends in the node mask
    for i in range(edge_index.shape[1]):
        iter_mask[i] = (edge_index[0, i] in node_numbers) and (
            edge_index[1, i] in node_numbers
        )

    return iter_mask

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


class Benzene(InMemoryComplexDataset):
    url = "https://github.com/mims-harvard/GraphXAI/raw/main/graphxai/datasets/real_world/benzene/benzene.npz"

    def __init__(self, root, max_ring_size, transform=None, pre_transform=None,include_down_adj=True, init_method='sum', n_jobs=1,):
        self.name = 'benzene'
        self._max_ring_size = max_ring_size
        self._use_edge_features = True
        self._n_jobs = n_jobs
        super().__init__(root, transform, pre_transform, max_dim=2,
                                              init_method=init_method, include_down_adj=include_down_adj,
                                              cellular=True)
        self.data, self.slices, idx = self.load_dataset()
        self.train_ids = idx['train']
        self.val_ids = idx['valid']
        self.test_ids = idx['test']

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def raw_file_names(self):
        return ['benzene.npz']

    @property
    def processed_dir(self):
        """This is overwritten, so the cellular complex data is placed in another folder"""
        prefix = "cell_" if self._cellular else ""
        directory = osp.join(self.root, self.name, f'{prefix}complex_dim{self.max_dim}_{self._init_method}')
        suffix1 = f"_{self._max_ring_size}rings" if self._cellular else ""
        suffix2 = "-E" if self._use_edge_features else ""
        return directory + suffix1 + suffix2

    @property
    def processed_file_names(self):
        return [f'{self.name}_complex.pt', f'{self.name}_idx.pt']

    def download(self):
        path = download_url(self.url, self.raw_dir)
    
    def load_dataset(self):
        print("Loading dataset from disk...")
        data, slices = torch.load(self.processed_paths[0])
        idx = torch.load(self.processed_paths[1])
        return data, slices, idx

    def process(self):
        data = np.load(self.raw_dir + "/benzene.npz", allow_pickle=True)

        data_list = []

        att, X, y, df = data["attr"], data["X"], data["y"], data["smiles"]
        ylist = [y[i][0] for i in range(y.shape[0])]
        X = X[0]

        for i in range(len(X)):
            x = torch.from_numpy(X[i]["nodes"])
            edge_attr = torch.from_numpy(X[i]["edges"])
            # y = X[i]['globals'][0]
            y = torch.tensor([int(ylist[i])], dtype=torch.long)

            # Get edge_index:
            e1 = torch.from_numpy(X[i]["receivers"]).long()
            e2 = torch.from_numpy(X[i]["senders"]).long()

            edge_index = torch.stack([e1, e2])

            # Get ground-truth explanation:
            node_imp = torch.from_numpy(att[i][0]["nodes"]).float()

            gt_node_label = torch.max(node_imp, dim=1)[0]

            # Error-check:
            assert (
                att[i][0]["n_edge"] == X[i]["n_edge"]
            ), "Num: {}, Edges different sizes".format(i)
            assert node_imp.shape[0] == x.shape[0], "Num: {}, Shapes: {} vs. {}".format(
                i, node_imp.shape[0], x.shape[0]
            ) + "\nExp: {} \nReal:{}".format(att[i][0], X[i])

            i_exps = []

            for j in range(node_imp.shape[1]):
                edge_imp = edge_mask_from_node_mask(
                    node_imp[:, j].bool(), edge_index=edge_index
                )
                i_exps.append(edge_imp)

            gt_edge_mask = torch.max(
                torch.stack([edge_imp for edge_imp in i_exps]), dim=0
            )[0]

            data_i = Data(
                x=x,
                y=y,
                edge_attr=edge_attr,
                edge_index=edge_index,
                edge_label=gt_edge_mask,
                node_label = gt_node_label
            )

            if self.pre_filter is not None and not self.pre_filter(data_i):
                continue

            if self.pre_transform is not None:
                data_i = self.pre_transform(data_i)

            data_list.append(data_i)

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