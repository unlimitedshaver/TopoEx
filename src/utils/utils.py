import os
import sys
import random
from tqdm import tqdm
from joblib import Parallel, delayed

import torch
import numpy as np

from .logger import log

def process_data(data):
    # if not use_edge_attr:
    #     data.edge_attr = None
    # if data.edge_label is None:
        # data.edge_label = torch.zeros(data.edge_index.shape[1])
    return data

def load_checkpoint(model, model_dir, model_name, map_location=None):
    checkpoint = torch.load(model_dir / (model_name + '.pt'), map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])


def save_checkpoint(model, model_dir, model_name):
    torch.save({'model_state_dict': model.state_dict()}, model_dir / (model_name + '.pt'))


init_metric_dict = {'metric/best_clf_epoch': 0, 'metric/best_clf_valid_loss': 0,
                    'metric/best_clf_train': 0, 'metric/best_clf_valid': 0, 'metric/best_clf_test': 0,
                    'metric/best_x_roc_train': 0, 'metric/best_x_roc_valid': 0, 'metric/best_x_roc_test': 0,
                    'metric/best_x_precision_train': 0, 'metric/best_x_precision_valid': 0, 'metric/best_x_precision_test': 0}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_cpu(tensor):
    return tensor.detach().cpu() if tensor is not None else None


def safe_index(l, e):
    try:
        return l.index(e)
    except:
        return len(l) - 1


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# def disable_rdkit_logging():
#     """
#     Disables RDKit whiny logging.
#     """
#     import rdkit.rdBase as rkrb
#     import rdkit.RDLogger as rkl
#     logger = rkl.logger()
#     logger.setLevel(rkl.ERROR)
#     rkrb.DisableLog('rdApp.error')


# def pmap_multi(pickleable_fn, data, n_jobs, verbose=1, desc=None, **kwargs):
#   results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None, )(
#     delayed(pickleable_fn)(*d, **kwargs) for i, d in tqdm(enumerate(data), desc=desc)
#   )

#   return results


def get_random_idx_split(dataset_len, split, seed):
    np.random.seed(seed)

    log('[INFO] Randomly split dataset!')
    idx = np.arange(dataset_len)
    np.random.shuffle(idx)

    n_train, n_valid = int(split['train'] * len(idx)), int(split['valid'] * len(idx))
    train_idx = idx[:n_train]
    valid_idx = idx[n_train:n_train+n_valid]
    test_idx = idx[n_train+n_valid:]
    return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

