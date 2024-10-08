import os
import torch
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.nn.functional as F


def to_item(tensor):
    if tensor is None:
        return None
    elif isinstance(tensor, torch.Tensor):
        return tensor.item()
    else:
        return tensor


def log_epoch(epoch, phase, loss_dict, log_dict, seed, writer, warmup, batch):
    if warmup:
        desc = f'[Seed {seed}, WramupEpoch: {epoch}]: {phase}....., ' if batch else f'[Seed {seed}, WramupEpoch: {epoch}]: {phase} done, '
    else:
        desc = f'[Seed {seed}, Epoch: {epoch}]: {phase}....., ' if batch else f'[Seed {seed}, Epoch: {epoch}]: {phase} done, '

    for k, v in loss_dict.items():
        if not batch and writer is not None:
            writer.add_scalar(f'warmup/{phase}/{k}', v, epoch) if warmup else writer.add_scalar(f'gsat_{phase}/{k}', v, epoch)
        desc += f'{k}: {v:.3f}, '

    eval_desc, org_clf_acc, org_clf_auc, masked_clf_acc, masked_clf_auc, exp_auc = get_eval_score(epoch, phase, log_dict, writer, warmup, batch)
    desc += eval_desc
    return desc, org_clf_acc, org_clf_auc, masked_clf_acc,exp_auc, masked_clf_auc,loss_dict['pred']


def get_eval_score(epoch, phase, log_dict, writer, warmup, batch):
    mul_class = False
    assert mul_class is False, 'multi-class not supported yet'

    record_dict = {}
    if warmup:
        if batch:
            record_dict['org_clf_logits'] = log_dict['org_clf_logits'][-1]      ##新加入log_dict,也即是当前处理的
            record_dict['clf_labels'] = log_dict['clf_labels'][-1]
        else:
            record_dict['org_clf_logits'] = torch.cat(log_dict['org_clf_logits'])
            record_dict['clf_labels'] = torch.cat(log_dict['clf_labels'])

        org_clf_preds = get_preds_from_logits(record_dict['org_clf_logits']) 
        clf_labels = record_dict['clf_labels']
        # org_clf_acc = (org_clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
        org_clf_acc = accuracy_score(clf_labels, org_clf_preds)
        desc = f'org_acc: {org_clf_acc:.3f}, '

        org_clf_auc = None
        if not batch:
            org_clf_auc = roc_auc_score(clf_labels, record_dict['org_clf_logits'].sigmoid()) if not mul_class else 0
            desc += f'org_auc: {org_clf_auc:.3f}, '
            if writer is not None:
                writer.add_scalar(f'warmup/{phase}/org_clf_acc', org_clf_acc, epoch)
                writer.add_scalar(f'warmup/{phase}/org_clf_auc', org_clf_auc, epoch)
        return [desc, org_clf_acc, org_clf_auc, *[-1]*2]

    # No warmup
    for k, v in log_dict.items():
        if batch:
            record_dict[k] = v[-1]
        else:
            record_dict[k] = torch.cat(v)

    org_clf_preds = get_preds_from_logits(record_dict['org_clf_logits'])
    clf_labels = record_dict['clf_labels']
    org_clf_acc = accuracy_score(clf_labels, org_clf_preds)

    masked_clf_preds = get_preds_from_logits(record_dict['masked_clf_logits'])
    # masked_clf_acc = (masked_clf_preds == clf_labels).sum().item() / clf_labels.shape[0]
    masked_clf_acc = accuracy_score(clf_labels, masked_clf_preds)

    desc = f'org_acc: {org_clf_acc:.3f}, msk_acc: {masked_clf_acc:.3f}, '
    if batch:
        return [desc, *[None]*5]

    try:
        org_clf_auc = roc_auc_score(clf_labels, record_dict['org_clf_logits'].sigmoid()) if not mul_class else 0
        desc += f'org_auc: {org_clf_auc:.3f}, '
    except ValueError:
        org_clf_auc = 0

    attn0 = record_dict['attn0']
    attn1 = record_dict['attn1']
    attn2 = record_dict['attn2']

    exp_labels = record_dict['exp_labels']
    # avg_auroc, angles, eigen_ratio = record_dict['avg_auroc'].mean(), record_dict['angles'].median(), record_dict['eigen_ratio'].median()
    # prec_at_k, prec_at_2k, prec_at_3k = record_dict['prec_at_k'].mean(), record_dict['prec_at_2k'].mean(), record_dict['prec_at_3k'].mean()

    try:
        masked_clf_auc = roc_auc_score(clf_labels, record_dict['masked_clf_logits'].sigmoid())  if not mul_class else 0
    except ValueError:
        masked_clf_auc = 0
    exp_auc = roc_auc_score(exp_labels, attn1)  ##
    bkg_attn_weights = attn1[exp_labels == 0]
    signal_attn_weights = attn1[exp_labels == 1]

    ##prec@k: {prec_at_k:.3f}, pred@2k: {prec_at_2k:.3f}, pred@3k: {prec_at_3k:.3f}
    desc += f'msk_auc: {masked_clf_auc:.3f}, exp_auc: {exp_auc:.3f}, ' + \
            f'bkg_attn: {bkg_attn_weights.mean():.3f}, sig_attn: {signal_attn_weights.mean():.3f}, '

    if writer is not None:
        writer.add_scalar(f'gsat_{phase}/org_clf_acc', org_clf_acc, epoch)
        writer.add_scalar(f'gsat_{phase}/org_clf_auc', org_clf_auc, epoch)
        writer.add_scalar(f'gsat_{phase}/masked_clf_acc', masked_clf_acc, epoch)
        writer.add_scalar(f'gsat_{phase}/masked_clf_auc', masked_clf_auc, epoch)
        writer.add_histogram(f'gsat_{phase}/attn0', attn0, epoch)
        writer.add_histogram(f'gsat_{phase}/attn1', attn1, epoch)
        writer.add_histogram(f'gsat_{phase}/attn2', attn2, epoch)
        writer.add_scalar(f'gsat_{phase}/exp_auc', exp_auc, epoch)
        # writer.add_scalar(f'gsat_{phase}/prec_at_k', prec_at_k, epoch)
        # writer.add_scalar(f'gsat_{phase}/prec_at_2k', prec_at_2k, epoch)
        # writer.add_scalar(f'gsat_{phase}/prec_at_3k', prec_at_3k, epoch)
        # writer.add_scalar(f'gsat_{phase}/avg_auroc', avg_auroc, epoch)

        writer.add_histogram(f'gsat_{phase}/bkg_attn_weights', bkg_attn_weights, epoch)
        writer.add_histogram(f'gsat_{phase}/signal_attn_weights', signal_attn_weights, epoch)

        writer.add_scalar(f'gsat_{phase}/avg_bkg_attn_weights/', bkg_attn_weights.mean(), epoch)
        writer.add_scalar(f'gsat_{phase}/avg_signal_attn_weights/', signal_attn_weights.mean(), epoch)
        writer.add_pr_curve(f'PR_Curve/gsat_{phase}/', exp_labels, attn1, epoch)

    return desc, org_clf_acc, org_clf_auc, masked_clf_acc, masked_clf_auc, exp_auc#, prec_at_k, prec_at_2k, prec_at_3k, angles, eigen_ratio


def get_preds_from_logits(logits):
    # multi-class
    # preds = np.argmax(logits, axis=1)
    preds = (logits.sigmoid() > 0.5).float()
    return preds


def get_prec_at_k(ids_of_ranked_attn, labels_for_graph_i, k):
    ids_of_topk_ranked_attn = ids_of_ranked_attn[:k]
    labels_of_topk_ranked_attn = labels_for_graph_i[ids_of_topk_ranked_attn]
    return (labels_of_topk_ranked_attn.sum().item() / k)


def get_precision_at_k_and_avgauroc_and_angles(exp_labels, attn, covar_mat, node_dir, topk, attn_graph_id):
    precision_at_k, precision_at_2k, precision_at_3k = [], [], []
    avg_auroc = []
    graph_ids = attn_graph_id.unique()
    for i in graph_ids:
        labels_for_graph_i = exp_labels[attn_graph_id == i]
        attn_for_graph_i = attn[attn_graph_id == i]
        covar_mat_for_graph_i = covar_mat[attn_graph_id == i] if covar_mat is not None else None
        node_dir_for_graph_i = node_dir[attn_graph_id == i] if node_dir is not None else None
        if labels_for_graph_i.sum() == 0:
            continue
        if labels_for_graph_i.sum() == len(labels_for_graph_i):
            continue

        ids_of_ranked_attn = np.argsort(-attn_for_graph_i)
        ids_of_topk_ranked_attn = ids_of_ranked_attn[:topk[0]]
        labels_of_topk_ranked_attn = labels_for_graph_i[ids_of_topk_ranked_attn]

        precision_at_k.append(get_prec_at_k(ids_of_ranked_attn, labels_for_graph_i, topk[0]))
        precision_at_2k.append(get_prec_at_k(ids_of_ranked_attn, labels_for_graph_i, topk[1]))
        precision_at_3k.append(get_prec_at_k(ids_of_ranked_attn, labels_for_graph_i, topk[2]))

        signal_nodes_by_thresholding = set(np.argsort(-attn_for_graph_i)[:np.argmax(np.diff(np.sort(-attn_for_graph_i)))+1].tolist())  # largest gap
        recalled_nodes = set(ids_of_topk_ranked_attn[labels_of_topk_ranked_attn == 1].tolist())  # top k
        selected_signal_nodes = sorted(list(signal_nodes_by_thresholding.intersection(recalled_nodes)))

        # avg_auroc.append(roc_auc_score(labels_for_graph_i, attn_for_graph_i))
        avg_auroc.append(0.0)  # to save time

    return torch.tensor(precision_at_k), torch.tensor(precision_at_2k), torch.tensor(precision_at_3k), torch.tensor(avg_auroc)


def log(*args):
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]', *args)


def update_and_save_best_epoch_res(baseline, train_res, valid_res, test_res, metric_dict, epoch, model_dir, seed, topk, warmup, writer):
    # assert len(train_res) == 11 ##这里不是11了
    main_metric_idx = 3 if not warmup else 1 # {1, 3} for AUC; {0, 2} for acc

    better_val_auc = valid_res[main_metric_idx] > metric_dict['metric/best_clf_valid']
    same_val_auc_but_better_val_loss = (valid_res[main_metric_idx] == metric_dict['metric/best_clf_valid']) and (valid_res[-1] < metric_dict['metric/best_clf_valid_loss'])

    # calc angle
    # if (better_val_auc or same_val_auc_but_better_val_loss) and (epoch > 300 or warmup):

    if better_val_auc or same_val_auc_but_better_val_loss:
        metric_dict = {'metric/best_clf_epoch': epoch, 'metric/best_clf_valid_loss': valid_res[-1],
                       'metric/best_clf_train': train_res[main_metric_idx], 'metric/best_clf_valid': valid_res[main_metric_idx], 'metric/best_clf_test': test_res[main_metric_idx],
                       'metric/best_x_roc_train': train_res[4], 'metric/best_x_roc_valid': valid_res[4], 'metric/best_x_roc_test': test_res[4],}
                    #    'metric/best_x_prec@k_train': train_res[5], 'metric/best_x_prec@k_valid': valid_res[5], 'metric/best_x_prec@k_test': test_res[5],
                    #    'metric/best_x_prec@2k_train': train_res[6], 'metric/best_x_prec@2k_valid': valid_res[6], 'metric/best_x_prec@2k_test': test_res[6],
                    #    'metric/best_x_prec@3k_train': train_res[7], 'metric/best_x_prec@3k_valid': valid_res[7], 'metric/best_x_prec@3k_test': test_res[7],

        if model_dir is not None:
            os.makedirs(model_dir, exist_ok=True)
            save_checkpoint(baseline, model_dir, model_name='model' )

    if model_dir is not None:
        os.makedirs(model_dir, exist_ok=True)
        save_checkpoint(baseline, model_dir, model_name=f'model_{epoch}')

    if writer is not None and not warmup:
        for metric, value in metric_dict.items():
            metric = metric.split('/')[-1]
            writer.add_scalar(f'best/{metric}', value, epoch)

    print(f'[Seed {seed}, Epoch: {epoch}]: Best Epoch: {metric_dict["metric/best_clf_epoch"]}, Best Val Pred Loss: {metric_dict["metric/best_clf_valid_loss"]:.3f}, '
            f'Best Val Pred AUROC: {metric_dict["metric/best_clf_valid"]:.3f}, Best Test Pred AUROC: {metric_dict["metric/best_clf_test"]:.3f}, '
            f'Best Test X AUROC: {metric_dict["metric/best_x_roc_test"]:.3f}, ')#Best Test X Prec@{topk[0]}: {metric_dict["metric/best_x_prec@k_test"]:.3f},
            # f'Best Test X Prec@{topk[1]}: {metric_dict["metric/best_x_prec@2k_test"]:.3f}, Best Test X Prec@{topk[2]}: {metric_dict["metric/best_x_prec@3k_test"]:.3f}, '
            # f'Best Test Angle: {metric_dict["metric/best_angle_test"]:.3f}, Best Test Eigen Ratio: {metric_dict["metric/best_eigen_test"]:.3f}')
    print('-' * 80), print('-' * 80)
    return metric_dict


def load_checkpoint(model, model_dir, model_name, map_location=None):
    checkpoint = torch.load(model_dir / (model_name + '.pt'), map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])


def save_checkpoint(model, model_dir, model_name):
    torch.save({'model_state_dict': model.state_dict()}, model_dir / (model_name + '.pt'))
