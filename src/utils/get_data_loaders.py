import torch
from torch_geometric.data import Data, Batch
from torch.utils.data.dataloader import default_collate
from torch._six import string_classes
import collections.abc as container_abcs


from pathlib import Path
from datasets import  PeptidesFunctionalDataset, SynGraphDataset, Mutag
from .cell_complex import Cochain, CochainBatch, Complex, ComplexBatch



class Collater(object):
    """Object that converts python lists of objects into the appropiate storage format.

    Args:
        follow_batch: Creates assignment batch vectors for each key in the list.
        max_dim: The maximum dimension of the cochains considered from the supplied list.
    """
    def __init__(self, follow_batch, max_dim=2):
        self.follow_batch = follow_batch
        self.max_dim = max_dim

    def collate(self, batch):
        """Converts a data list in the right storage format."""
        elem = batch[0]
        if isinstance(elem, Cochain):
            return CochainBatch.from_cochain_list(batch, self.follow_batch)
        elif isinstance(elem, Complex):
            return ComplexBatch.from_complex_list(batch, self.follow_batch, max_dim=self.max_dim)
        elif isinstance(elem, Data):
            return Batch.from_data_list(batch, self.follow_batch)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)


class DataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges cochain complexes into to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
        max_dim (int): The maximum dimension of the chains to be used in the batch.
            (default: 2)
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
                 max_dim=2, **kwargs):

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for Pytorch Lightning...
        self.follow_batch = follow_batch

        super(DataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=Collater(follow_batch, max_dim), **kwargs)


def get_data_loaders(dataset_name, batch_size, data_config, seed=42):
    data_dir = Path(data_config['data_dir'])
    # assert dataset_name in ['tau3mu', 'plbind', 'synmol'] or 'acts' in dataset_name

    # elif dataset_name == 'tau3mu':
    #     dataset = Tau3Mu(data_dir / 'tau3mu', data_config=data_config, seed=seed)
    #     loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, idx_split=dataset.idx_split)

    if dataset_name == 'peptides_f':
        dataset = PeptidesFunctionalDataset(data_dir / 'peptides_f', max_ring_size=data_config['max_ring_size'])
        loaders, test_set= get_loaders_and_test_set(batch_size, dataset=dataset)
    elif dataset_name == 'ba_2motifs':
        dataset = SynGraphDataset(data_dir, name='ba_2motifs', max_ring_size=data_config['max_ring_size'])
        loaders, test_set= get_loaders_and_test_set(batch_size, dataset=dataset)
    elif dataset_name == 'mutag':
        dataset = Mutag(data_dir, max_ring_size=data_config['max_ring_size'])
        loaders, test_set= get_loaders_and_test_set(batch_size, dataset=dataset)

    x_dim = test_set[0].nodes.x.shape[1]
    edge_attr_dim = 0 if test_set[0].edges is None else test_set[0].edges.x.shape[1]

    return loaders, dataset, test_set, x_dim, edge_attr_dim


def get_loaders_and_test_set(batch_size, dataset):
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset.get_split('train'), batch_size=batch_size,
            shuffle=True, num_workers=0, max_dim=dataset.max_dim)
    valid_loader = DataLoader(dataset.get_split('valid'), batch_size=batch_size,
            shuffle=False, num_workers=0, max_dim=dataset.max_dim)
    test_split = split_idx.get("test", None)
    test_loader = None
    if test_split is not None:
        test_loader = DataLoader(dataset.get_split('test'), batch_size=batch_size,
                shuffle=False, num_workers=0, max_dim=dataset.max_dim)

    test_set = dataset.copy(dataset.test_ids)  # For visualization
    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}, test_set
