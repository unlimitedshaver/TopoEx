import yaml
import json
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import logging


import networkx as nx
# from rdkit.Chem import PeriodicTable  ##conda install -c   ##pip install rdkit
from rdkit import Chem
import matplotlib.pyplot as plt

###
from torch_geometric.data import Data
from utils.get_data_loaders import DataLoader
from torch_geometric.utils import subgraph, to_networkx
###

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.cell_complex import ComplexBatch

# from get_model import Model
from baselines import LRIBern, LRIGaussian, Grad, BernMaskP, BernMask, PointMask
from utils import to_cpu, log_epoch, get_data_loaders, process_data, set_seed, init_metric_dict, update_and_save_best_epoch_res, load_checkpoint, ExtractorMLP, get_optimizer, process_data

from backbones.cinpp import OGBEmbedCINpp


def eval_one_batch(baseline, optimizer, data, epoch, warmup, phase, method_name):
    with torch.set_grad_enabled(method_name in ['gradcam', 'gradgeo', 'bernmask']):
        assert optimizer is None
        baseline.extractor.eval() if hasattr(baseline, 'extractor') else None
        baseline.clf.eval()

        # calc angle
        # do_sampling = True if phase == 'valid' and method_name == 'lri_gaussian' else False

        # BernMaskP
        do_sampling = False
        # do_sampling = True if phase == 'valid' and method_name == 'bernmask_p' else False # we find this is better for BernMaskP
        loss, loss_dict, org_clf_logits, masked_clf_logits, node_attn, edge_attn, cell_attn = baseline.forward_pass(data, epoch, warmup=warmup, do_sampling=do_sampling)
        return loss_dict, to_cpu(org_clf_logits), to_cpu(masked_clf_logits), to_cpu(node_attn), to_cpu(edge_attn), to_cpu(cell_attn)


def train_one_batch(baseline, optimizer, data, epoch, warmup, phase, method_name):
    baseline.extractor.train() if hasattr(baseline, 'extractor') else None
    baseline.clf.train() if (method_name != 'bernmask_p' or warmup) else baseline.clf.eval()

    loss, loss_dict, org_clf_logits, masked_clf_logits, node_attn, edge_attn, cell_attn = baseline.forward_pass(data, epoch, warmup=warmup, do_sampling=True)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss_dict, to_cpu(org_clf_logits), to_cpu(masked_clf_logits), to_cpu(node_attn), to_cpu(edge_attn), to_cpu(cell_attn)


def run_one_epoch(baseline, optimizer, data_loader, epoch, phase, warmup, seed, device, writer, method_name):
    loader_len = len(data_loader)
    run_one_batch = train_one_batch if phase == 'train' else eval_one_batch
    phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar
    log_dict = {k: [] for k in ['attn0', 'attn1', 'attn2', 'clf_labels', 'org_clf_logits', 'masked_clf_logits', 'exp_labels']}
    all_loss_dict = {}

    num_skips = 0

    pbar = tqdm(data_loader)
    for idx, data in enumerate(pbar):
        data = process_data(data)   
        data = data.to(device)
        if isinstance(data, ComplexBatch):
            num_samples = data.cochains[0].x.size(0)
            for dim in range(1, data.dimension+1):
                num_samples = min(num_samples, data.cochains[dim].num_cells)
        else:
            num_samples = data.x.size(0)
        
        if num_samples <= 1:
            # Skip batch if it only comprises one sample (could cause problems with BN)
            num_skips += 1
            if float(num_skips) / loader_len >= 0.25:
                logging.warning("Warning! 25% of the batches were skipped this epoch")
            continue

        loss_dict, org_clf_logits, masked_clf_logits, attn0, attn1, attn2= run_one_batch(baseline, optimizer, data, epoch, warmup, phase, method_name)
        clf_labels = to_cpu(data.y)
        exp_labels = to_cpu(data.edge_label)
        # precision_at_k = self.get_precision_at_k(att, exp_labels, self.k, data.batch, data.edge_index)
        # avg_auroc = []

        # if not warmup:
        #     attn0, attn1, attn2 = get_relevant_nodes(attn0, attn1, attn2, data.cochains, data.y, signal_class)
            # prec_at_k, prec_at_2k, prec_at_3k, avg_auroc, angles,  _, eigen_ratio, _ = get_precision_at_k_and_avgauroc_and_angles(exp_labels, attn, covar_mat, node_dir, topk, attn_graph_id)
            # avg_auroc = []
            # avg_auroc.append(0.0)
            # avg_auroc = torch.tensor(avg_auroc)


        for key in log_dict.keys():
            if eval(key) is not None or warmup:
                log_dict[key].append(eval(key))

        desc = log_epoch(epoch, phase, loss_dict, log_dict, seed, writer, warmup, batch=True)[0]
        for k, v in loss_dict.items():
            all_loss_dict[k] = all_loss_dict.get(k, 0) + v

        if idx == loader_len - 1:
            for k, v in all_loss_dict.items():
                all_loss_dict[k] = v / loader_len
            desc, org_clf_acc, org_clf_auc, masked_clf_acc, masked_clf_auc, exp_auc, avg_loss = log_epoch(epoch, phase, all_loss_dict, log_dict, seed, writer, warmup, batch=False)
        pbar.set_description(desc)
    return org_clf_acc, org_clf_auc, masked_clf_acc, masked_clf_auc, exp_auc, avg_loss


def get_relevant_nodes(attn0, attn1, attn2, attn_graph_id, y, signal_class):
    if signal_class is not None:
        y = y[:, signal_class]
        attn_c0 = attn_graph_id[0].batch.reshape(-1)
        attn_c1 = attn_graph_id[1].batch.reshape(-1)
        attn_c2 = attn_graph_id[2].batch.reshape(-1)
        in_signal_class_0 = (y[attn_c0] == 1).reshape(-1)
        in_signal_class_0 = to_cpu(in_signal_class_0)
        in_signal_class_1 = (y[attn_c1] == 1).reshape(-1)
        in_signal_class_1 = to_cpu(in_signal_class_1)
        in_signal_class_2 = (y[attn_c2] == 1).reshape(-1)
        in_signal_class_2 = to_cpu(in_signal_class_2)
        attn0, attn1, attn2 =  attn0[in_signal_class_0], attn1[in_signal_class_1], attn2[in_signal_class_2]
        # if node_dir is not None:
        #     node_dir = node_dir[in_signal_class]
        # if covar_mat is not None:
        #     covar_mat = covar_mat[in_signal_class]
    return attn0, attn1, attn2


def train(config, method_name, model_name, seed, dataset_name, log_dir, device):
    writer = SummaryWriter(log_dir) if log_dir is not None else None
    topk = config['logging']['topk']

    batch_size = config['optimizer']['batch_size']
    epochs = config[method_name]['epochs']
    warmup = config[method_name]['warmup']
    data_config = config['data']
    model_config = config['model'][model_name]
    loaders, dataset, test_set, x_dim, edge_attr_dim = get_data_loaders(dataset_name, batch_size, data_config, seed)
    signal_class = None  #dataset.signal_class  1  

    # clf = Model(model_name, config['model'][model_name], method_name, config[method_name], dataset).to(device)
    clf = OGBEmbedCINpp(x_dim,
                        edge_attr_dim,
                        1,                       # out_size
                        model_config['num_layers'],                         # num_layers
                        model_config['emb_dim'],                            # hidden
                        dropout_rate=model_config['drop_rate'],             # dropout_rate
                        indropout_rate=0.0,         # in-dropout_rate
                        max_dim=dataset.max_dim,                 # max_dim
                        jump_mode=None,                # jump_mode
                        nonlinearity=model_config['nonlinearity'],          # nonlinearity
                        readout=model_config['readout'],                    # readout
                        final_readout=model_config['final_readout'],        # final readout
                        apply_dropout_before=model_config['drop_position'], # where to apply dropout
                        use_coboundaries=True,       # whether to use coboundaries
                        embed_edge=True,       # whether to use edge feats
                        graph_norm=model_config['graph_norm'],              # normalization layer
                        readout_dims=(0, 1, 2),              # readout_dims
                        atom_encoder=model_config["atom_encoder"]
                        ).to(device)
    extractor = ExtractorMLP(model_config['emb_dim'], config[method_name])
    extractor = extractor.to(device)
    criterion = F.binary_cross_entropy_with_logits

    if method_name == 'lri_bern':
        baseline = LRIBern(clf, extractor, criterion, config['lri_bern'])
    elif method_name == 'lri_gaussian':
        baseline = LRIGaussian(clf, extractor, criterion, config['lri_gaussian'])
    elif method_name == 'gradgeo':
        baseline = Grad(clf, signal_class, criterion, config['gradgeo'])
    elif method_name == 'gradcam':
        baseline = Grad(clf, signal_class, criterion, config['gradcam'])
    elif method_name == 'bernmask':
        baseline = BernMask(clf, extractor, criterion, config['bernmask'])
    elif method_name == 'bernmask_p':
        baseline = BernMaskP(clf, extractor, criterion, config['bernmask_p'])
    elif method_name == 'pointmask':
        baseline = PointMask(clf, extractor, criterion, config['pointmask'])
    else:
        raise ValueError('Unknown method: {}'.format(method_name))

    optimizer = get_optimizer(clf, extractor, config['optimizer'], config[method_name], warmup=True, slayer= baseline.readout)
    metric_dict = deepcopy(init_metric_dict)
    for epoch in range(warmup):
        train_res = run_one_epoch(baseline, optimizer, loaders['train'], epoch, 'train', warmup, seed, device, writer, method_name)
        valid_res = run_one_epoch(baseline, None, loaders['valid'], epoch, 'valid', warmup, seed, device, writer, method_name)
        test_res = run_one_epoch(baseline, None, loaders['test'], epoch, 'test', warmup, seed, device, writer, method_name)
        metric_dict = update_and_save_best_epoch_res(baseline, train_res, valid_res, test_res, metric_dict, epoch, log_dir, seed, topk, True, writer)

    # if method_name in ['gradcam', 'gradgeo', 'bernmask_p', 'bernmask']:
    #     load_checkpoint(baseline, log_dir, model_name='wp_model')
    #     if 'grad' in method_name: baseline.start_tracking()

    warmup = 0
    metric_dict = deepcopy(init_metric_dict)
    clf.emb_model = deepcopy(clf.model) if not config[method_name].get('one_encoder', True) else None
    optimizer = get_optimizer(clf, extractor, config['optimizer'], config[method_name], warmup=False, slayer= baseline.readout)
    for epoch in range(epochs):
        if method_name in ['gradcam', 'gradgeo', 'bernmask']:
            if method_name == 'bernmask':
                train_res = None
            else:
                train_res = run_one_epoch(baseline, None, loaders['train'], epoch, 'test', warmup, seed, device, writer,  method_name)
            valid_res = run_one_epoch(baseline, None, loaders['valid'], epoch, 'test', warmup, seed,  device, writer,  method_name)
            test_res = run_one_epoch(baseline, None, loaders['test'], epoch, 'test', warmup, seed, device,  writer,  method_name)
            if train_res is None:
                train_res = valid_res
        else:
            train_res = run_one_epoch(baseline, optimizer, loaders['train'], epoch, 'train', warmup, seed, device, writer,  method_name)
            valid_res = run_one_epoch(baseline, None, loaders['valid'], epoch, 'valid', warmup, seed, device, writer,  method_name)
            test_res = run_one_epoch(baseline, None, loaders['test'], epoch, 'test', warmup, seed, device,  writer, method_name)

        metric_dict = update_and_save_best_epoch_res(baseline, train_res, valid_res, test_res, metric_dict, epoch, log_dir, seed, topk, False, writer)
        report_dict = {k.replace('metric/best_', ''): v for k, v in metric_dict.items()}  # for better readability
    return report_dict


def run_one_seed(dataset_name, method_name, model_name, cuda_id, seed, note, time):
    set_seed(seed)
    config_name = dataset_name  #.split('_')[0]
    # sub_dataset_name = '_' + dataset_name.split('_')[1] if len(dataset_name.split('_')) > 1 else ''
    config_path = Path('./configs') /  f'{config_name}.yml'
    config = yaml.safe_load((config_path).open('r'))
    if config[method_name].get(model_name, False):
        config[method_name].update(config[method_name][model_name])
    print('-' * 80), print('-' * 80)
    print(f'Config: ', json.dumps(config, indent=4))

    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')
    log_dir = None
    if config['logging']['tensorboard'] or method_name in ['gradcam', 'gradgeo', 'bernmask_p', 'bernmask']:
        log_dir = Path(config['data']['data_dir']) / config_name / ('-'.join([time, method_name, model_name, 'seed'+str(seed), note]))
        log_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(config_path, log_dir / config_path.name)
    report_dict = train(config, method_name, model_name, seed, dataset_name, log_dir, device)
    return report_dict


def main():
    time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S.%f")[:-3]
    parser = argparse.ArgumentParser(description='Train SAT')
    parser.add_argument('-d', '--dataset', type=str, help='dataset used', default='ba_2motifs')
    parser.add_argument('-m', '--method', type=str, help='method used', default='lri_bern')
    parser.add_argument('-b', '--backbone', type=str, help='backbone used', default='cinpp')
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu', default=1)
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument('--note', type=str, help='note in log name', default='')
    args = parser.parse_args()

    print(args)
    report_dict = run_one_seed(args.dataset, args.method, args.backbone, args.cuda, args.seed, args.note, time)
    print(json.dumps(report_dict, indent=4))

## viz
def get_viz_idx(test_set, dataset_name, num_viz_samples):
    y_dist = test_set.data['labels'].numpy().reshape(-1)
    num_nodes = np.array([each.cochains[0].x.shape[0] for each in test_set])
    classes = np.unique(y_dist)
    res = []
    for each_class in classes:
        tag = 'class_' + str(each_class)
        if dataset_name == 'Graph-SST2':
            condi = (y_dist == each_class) * (num_nodes > 5) * (num_nodes < 10)  # in case too short or too long
            candidate_set = np.nonzero(condi)[0]
        else:
            candidate_set = np.nonzero(y_dist == each_class)[0]
        idx = np.random.choice(candidate_set, num_viz_samples, replace=False)
        res.append((idx, tag))

    if dataset_name == 'mutag':
        for each_class in classes:
            tag = 'class_' + str(each_class)
            candidate_set = np.nonzero(y_dist == each_class)[0]
            idx = np.random.choice(candidate_set, num_viz_samples, replace=False)
            res.append((idx, tag))
    return res


def visualize_results(gsat, all_viz_set, test_set, num_viz_samples, dataset_name, optimizer, method_name):
    figsize = 10
    fig, axes = plt.subplots(len(all_viz_set), num_viz_samples, figsize=(figsize*num_viz_samples, figsize*len(all_viz_set)*0.8))

    for class_idx, (idx, tag) in enumerate(all_viz_set):
        viz_set = test_set[idx]
        data = next(iter(DataLoader(viz_set, batch_size=len(idx), shuffle=False)))
        # data = process_data(data, use_edge_attr)
        _, _, _, _, batch_att, _ = eval_one_batch(gsat, optimizer, data.to(gsat.device), 500, False, None, method_name)
        #baseline, optimizer, data, epoch, warmup, phase, method_name

        for i in tqdm(range(len(viz_set))):
            mol_type, coor = None, None
            if dataset_name == 'mutag':
                node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
                mol_type = {k: node_dict[v.item()] for k, v in enumerate(viz_set[i]._stores['node_type'])}
            elif dataset_name == 'Graph-SST2':
                mol_type = {k: v for k, v in enumerate(viz_set[i].sentence_tokens)}
                num_nodes = data.x.shape[0]
                x = np.linspace(0, 1, num_nodes)
                y = np.ones_like(x)
                coor = np.stack([x, y], axis=1)
            elif dataset_name == 'ogbg_molhiv':
                element_idxs = {k: int(v+1) for k, v in enumerate(viz_set[i].x[:, 0])}
                mol_type = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v)) for k, v in element_idxs.items()}
            elif dataset_name == 'mnist':
                raise NotImplementedError

            node_subset = data.cochains[0].batch == i
            ##
            second_row = data.cochains[1].boundary_index[1]
            res = {}
            for ix in range(second_row.shape[0]):
                key = second_row[ix].item()
                if key not in res:
                    res[key] = []
                res[key].append(data.cochains[1].boundary_index[:,ix])
            new_index = {}
            for k,v in res.items():
                assert len(v) == 2
                key1 = (v[0][0].item(), v[1][0].item())
                new_index[key1] = batch_att[k]
                key2 = (v[1][0].item(), v[0][0].item()) 
                new_index[key2] = batch_att[k]
            edges = data.cochains[0].upper_index
            batch_attn = []
            for e in range(edges.shape[1]):
                edge = (edges[0][e].item(), edges[1][e].item())
                batch_attn.append(new_index[edge])
            ##
            _, edge_mask = subgraph(node_subset.cpu(), data.cochains[0].upper_index.cpu(), edge_attr=torch.tensor(batch_attn))#edge_attr=batch_att

            node_label = viz_set[i].node_label.reshape(-1) #if viz_set[i].get('node_label', None) is not None else torch.zeros(viz_set[i].x.shape[0])
            visualize_a_graph(viz_set[i].cochains[0].upper_index, edge_mask, node_label, dataset_name, axes[class_idx, i], norm=True, mol_type=mol_type, coor=coor)
            # axes[class_idx, i].axis('off')
        fig.tight_layout()

    each_plot_len = 1/len(viz_set)
    for num in range(1, len(viz_set)):
        line = plt.Line2D((each_plot_len*num, each_plot_len*num), (0, 1), color="gray", linewidth=1, linestyle='dashed', dashes=(5, 10))
        fig.add_artist(line)

    each_plot_width = 1/len(all_viz_set)
    for num in range(1, len(all_viz_set)):
        line = plt.Line2D((0, 1), (each_plot_width*num, each_plot_width*num), color="gray", linestyle='dashed', dashes=(5, 10))
        fig.add_artist(line)


def visualize_a_graph(edge_index, edge_att, node_label, dataset_name, ax, coor=None, norm=False, mol_type=None, nodesize=300):
    if norm:  # for better visualization
        edge_att = edge_att*10**5
        edge_att = (edge_att - edge_att.min()) / (edge_att.max() - edge_att.min() + 1e-6)

    if mol_type is None or dataset_name == 'Graph-SST2':
        atom_colors = {0: '#E49D1C', 1: '#FF5357', 2: '#a1c569', 3: '#69c5ba'}
        node_colors = [None for _ in range(node_label.shape[0])]
        for y_idx in range(node_label.shape[0]):
            node_colors[y_idx] = atom_colors[node_label[y_idx].int().tolist()]
    else:
        node_color = ['#29A329', 'lime', '#F0EA00',  'maroon', 'brown', '#E49D1C', '#4970C6', '#FF5357']
        element_idxs = {k: Chem.PeriodicTable.GetAtomicNumber(Chem.GetPeriodicTable(), v) for k, v in mol_type.items()}
        node_colors = [node_color[(v - 1) % len(node_color)] for k, v in element_idxs.items()]

    data = Data(edge_index=edge_index, att=edge_att, y=node_label, num_nodes=node_label.size(0)).to('cpu')
    G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])

    # calculate Graph positions
    if coor is None:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = {idx: each.tolist() for idx, each in enumerate(coor)}

    for source, target, data in G.edges(data=True):
        ax.annotate(
            '', xy=pos[target], xycoords='data', xytext=pos[source],
            textcoords='data', arrowprops=dict(
                arrowstyle="->" if dataset_name == 'Graph-SST2' else '-',
                lw=max(data['att'], 0) * 3,
                alpha=max(data['att'], 0),  # alpha control transparency
                color='black',  # color control color
                shrinkA=np.sqrt(nodesize) / 2.0 + 1,
                shrinkB=np.sqrt(nodesize) / 2.0 + 1,
                connectionstyle='arc3,rad=0.4' if dataset_name == 'Graph-SST2' else 'arc3'
            ))

    if mol_type is not None:
        nx.draw_networkx_labels(G, pos, mol_type, ax=ax)

    if dataset_name != 'Graph-SST2':
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=nodesize, ax=ax)
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', arrows=False, alpha=0.1, ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', arrows=False, alpha=0.1, ax=ax, connectionstyle='arc3,rad=0.4')



if __name__ == '__main__':
    main()




