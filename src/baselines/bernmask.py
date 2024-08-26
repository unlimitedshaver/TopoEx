# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/gnn_explainer.html
# https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py

import torch
import torch.nn as nn
from math import sqrt


class BernMask(nn.Module):

    def __init__(self, clf, extractor, criterion, config):
        super().__init__()
        self.clf = clf
        self.extractor = extractor
        self.criterion = criterion
        self.device = next(self.parameters()).device

        self.pred_loss_coef = config['pred_loss_coef']
        self.size_loss_coef = config['size_loss_coef']
        self.mask_ent_loss_coef = config['mask_ent_loss_coef']
        self.iter_per_sample = config['iter_per_sample']
        self.iter_lr = config['iter_lr']

    def __loss__(self, node_mask, edge_mask, cell_mask, clf_logits, clf_labels, epoch, warmup):
        pred_loss = self.criterion(clf_logits, clf_labels.float())
        if warmup:
            return pred_loss, {'loss': pred_loss.item(), 'pred': pred_loss.item()}

        v_size_loss = node_mask.mean()
        v_mask_ent_reg = -node_mask * (node_mask + 1e-10).log() - (1 - node_mask) * (1 - node_mask + 1e-10).log()
        v_mask_ent_loss = v_mask_ent_reg.mean()

        e_size_loss = edge_mask.mean()
        e_mask_ent_reg = -edge_mask * (edge_mask + 1e-10).log() - (1 - edge_mask) * (1 - edge_mask + 1e-10).log()
        e_mask_ent_loss = e_mask_ent_reg.mean()

        c_size_loss = cell_mask.mean()
        c_mask_ent_reg = -cell_mask * (cell_mask + 1e-10).log() - (1 - cell_mask) * (1 - cell_mask + 1e-10).log()
        c_mask_ent_loss = c_mask_ent_reg.mean()

        pred_loss = self.pred_loss_coef * pred_loss
        size_loss = self.size_loss_coef * v_size_loss + self.size_loss_coef * e_size_loss + self.size_loss_coef * c_size_loss
        mask_ent_loss = self.mask_ent_loss_coef * v_mask_ent_loss + self.mask_ent_loss_coef * e_mask_ent_loss + self.mask_ent_loss_coef * c_mask_ent_loss

        loss = pred_loss + size_loss + mask_ent_loss
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'size': size_loss.item(), 'ent': mask_ent_loss.item()}
        return loss, loss_dict

    def _initialize_masks(self, emb, init="normal"):
        N_v = emb[0].size(0)
        N_e = emb[1].size(0)
        N_c = emb[2].size(0)

        std = torch.nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * N_v))
        self.node_mask = torch.nn.Parameter(torch.FloatTensor(N_v, 1).normal_(1, std))

        std = torch.nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * N_e))
        self.edge_mask = torch.nn.Parameter(torch.FloatTensor(N_e, 1).normal_(1, std))

        
        std = torch.nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * N_c))
        self.cell_mask = torch.nn.Parameter(torch.FloatTensor(N_c, 1).normal_(1, std))


    def _clear_masks(self):
        self.node_mask = None

    def forward_pass(self, data, epoch, warmup, **kwargs):
        if warmup:
            clf_logits = self.clf(data)
            loss, loss_dict = self.__loss__(None, None, None, clf_logits, data.y, epoch, warmup)
            return loss, loss_dict, clf_logits, None, None, None, None

        self._clear_masks()
        emb = self.clf.get_emb(data)
        params = data.get_all_cochain_params(max_dim=2, include_down_features=True)
        v_index = [None, params[0].up_index, None]        
        e_index = [params[1].boundary_index, params[1].up_index, params[1].down_index]
        c_index = [params[2].boundary_index, None, params[2].down_index]
        with torch.no_grad():
            original_clf_logits = self.clf(data)

        self._initialize_masks(emb)
        # self.to(data.x.device)
        self.to(data.cochains[0].x.device)

        parameters = [self.node_mask, self.edge_mask, self.cell_mask]
        
        optimizer = torch.optim.Adam(parameters, lr=self.iter_lr)

        for _ in range(self.iter_per_sample):
            optimizer.zero_grad()

            node_mask = torch.sigmoid(self.node_mask)
            edge_mask = torch.sigmoid(self.edge_mask)
            cell_mask = torch.sigmoid(self.cell_mask)
        
            # edge_mask = self.node_attn_to_edge_attn(node_mask, edge_index)
            bon_attn = [None, self.node_attn_to_edge_attn(node_mask,edge_mask, e_index[0]), self.node_attn_to_edge_attn(edge_mask, cell_mask, c_index[0])]
            up_attn = [self.node_attn_to_edge_attn(node_mask,node_mask, v_index[1]), self.node_attn_to_edge_attn(edge_mask,edge_mask, e_index[1]), None]
            down_attn = [None, self.node_attn_to_edge_attn(edge_mask,edge_mask, e_index[2]), self.node_attn_to_edge_attn(cell_mask,cell_mask, c_index[2])]
            
            masked_clf_logits = self.clf(data, up_attn = up_attn, down_attn = down_attn, bon_attn = bon_attn)

            loss, loss_dict = self.__loss__(self.node_mask.sigmoid(), self.edge_mask.sigmoid(), self.cell_mask.sigmoid(), masked_clf_logits, original_clf_logits.sigmoid(), epoch, warmup)
            loss.backward()
            optimizer.step()

        return loss, loss_dict, original_clf_logits, masked_clf_logits, self.node_mask.sigmoid().reshape(-1), self.edge_mask.sigmoid().reshape(-1), self.cell_mask.sigmoid().reshape(-1)

    @staticmethod
    def node_attn_to_edge_attn(src_attn, dst_attn, edge_index):
        if edge_index is None:
            return None
        src_attn = src_attn[edge_index[0]]
        dst_attn = dst_attn[edge_index[1]]
        edge_attn = src_attn * dst_attn
        return edge_attn
