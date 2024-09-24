import torch
import torch.nn as nn
from torch_scatter import scatter
import numpy as np

from torchPHext.torchex_PHext import pershom as pershom_ext
# ph_extended_link_tree = pershom_ext.pershom_backend.__C.VertExtendedFiltCompCuda_link_cut_tree__extended_persistence_batch
ph_extended_link_tree_cyclereps= pershom_ext.pershom_backend.__C.VertExtendedFiltCompCuda_link_cut_tree_cyclereps__extended_persistence_batch

from utils import PershomReadout, GaussianMixtureModel


class LRIBern(nn.Module):

    def __init__(self, clf, extractor, criterion, config):
        super().__init__()
        self.clf = clf
        self.extractor = extractor
        self.criterion = criterion
        self.device = next(self.parameters()).device

        self.pred_loss_coef = config['pred_loss_coef']
        self.info_loss_coef = config['info_loss_coef']
        self.temperature = config['temperature']

        # self.final_r = config['final_r']
        self.decay_interval = config['decay_interval']
        self.decay_r = config['decay_r']
        self.init_r = config['init_r']

        self.attn_constraint = config['attn_constraint']

        ## for gaus
        self.gaus = GaussianMixtureModel()

        ## for ph
        self.readout = PershomReadout(
            # dataset,
            num_struct_elements=16,
            # cls_hidden_dimension=cls_hidden_dimension,
            # drop_out=drop_out
        ).to(self.device)
 
    def compute_extended_ph_link_tree_wcyclereps(self,node_filt, batch, device):
        ph_input= []
        for idx, (i, j, e) in enumerate(zip(batch.sample_pos[:-1], batch.sample_pos[1:], batch.boundary_up)):
            v = node_filt[i:j]# extract vertex values
            v = v.squeeze()
            v = v.cpu()
            v = 1 - v   ## Upper Filtration || Lower Filtration
            e = e.cpu()
            ph_input.append((v, [e]))
        # ph_input needs to be: (v,[e])
        out= ph_extended_link_tree_cyclereps(ph_input)
        pers= [per[0] for per in out]
        cycle_reps = [cycles[1] for cycles in out]

        h_0_up= [torch.stack([x.to(device) for x in g[0]]) for g in pers]
        h_0_down= [torch.stack([x.to(device) for x in g[1]]) for g in pers]
        h_0_extplus= [torch.stack([x.to(device) for x in g[2]]) for g in pers]
        h_1_extminus= [torch.stack([x.to(device) for x in g[3]]) for g in pers]
        cycle_reps= [[torch.stack([x.to(device).unsqueeze(0) for x in c]) for c in cycle] for cycle in cycle_reps]  

        return h_0_up, h_0_down, h_0_extplus, h_1_extminus, cycle_reps

    def __loss__(self, ver_attn, edge_attn, cell_attn, clf_logits, clf_labels, epoch, warmup, tpl, gas):
        pred_loss = self.criterion(clf_logits, clf_labels.float().view(clf_logits.shape))          
        if warmup:
            return pred_loss, {'loss': pred_loss.item(), 'pred': pred_loss.item()}

        # r_0 = self.get_r(epoch, 0.7)
        # r_1 = self.get_r(epoch, 0.7)
        # r_2 = self.get_r(epoch, 0.7)
        # v_info_loss = (ver_attn * torch.log(ver_attn/r_0 + 1e-6) + (1 - ver_attn) * torch.log((1 - ver_attn)/(1 - r_0 + 1e-6) + 1e-6)).mean()

        pred_loss = self.pred_loss_coef * pred_loss
        info_loss =  0#self.info_loss_coef * e_info_loss + self.info_loss_coef * c_info_loss + self.info_loss_coef * v_info_loss

        loss = pred_loss + info_loss + 0.01*tpl + 0.1*gas
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'v_info': 0, 'gas': gas.item(),'c_info': 0, 'tpl':tpl}
        return loss, loss_dict

    def forward_pass(self, data, epoch, warmup, do_sampling):
        if warmup:
            clf_logits = self.clf(data)
            loss, loss_dict = self.__loss__(None, None, None, clf_logits, data.y, epoch, warmup, None)
            return loss, loss_dict, clf_logits, None, None, None, None

        emb, graph_x = self.clf.get_emb(data)
        cell_attn_log_logits = self.extractor(emb)

        ## obtain all the adjacency matrices
        params = data.get_all_cochain_params(max_dim=2, include_down_features=True)
        v_index = [None, params[0].up_index, None]        
        e_index = [params[1].boundary_index, params[1].up_index, params[1].down_index]
        c_index = [params[2].boundary_index, None, params[2].down_index]

        # if self.attn_constraint == 'smooth_min':
        #     node_attn_log_logits = scatter(node_attn_log_logits[edge_index[1]].reshape(-1), edge_index[0], reduce='min').reshape(-1, 1)
        #     node_attn_log_logits = scatter(node_attn_log_logits[edge_index[1]].reshape(-1), edge_index[0], reduce='min').reshape(-1, 1)
        # else:
        #     assert self.attn_constraint == 'none'

        ver_attn = self.sampling(cell_attn_log_logits[0], do_sampling)
        edge_attn = self.sampling(cell_attn_log_logits[1], do_sampling)
        cell_attn = self.sampling(cell_attn_log_logits[2], do_sampling)

        ## norm
        ver_attn = self.min_max_normalize(ver_attn)
        edge_attn = self.min_max_normalize(edge_attn)
        cell_attn = torch.ones_like(cell_attn, dtype=torch.float32)
        
        ## gaus
        gas = self.gaus(edge_attn)

        ## ph
        beta_0_up, beta_0_down, beta0_ext, beta1_ext, cyl = self.compute_extended_ph_link_tree_wcyclereps(ver_attn, data, self.device)
        ph_x, tpl = self.readout(beta_0_up, beta_0_down, beta0_ext, beta1_ext, graph_x, cyl)# [ver,ver] [ver, edge] [edge, edge] H0, H1 

        ## control experiment
        original_clf_logits = self.clf(data, up_attn = [None,None,None], down_attn = [None,None,None], bon_attn = [None,None,None], ph_x = ph_x)
        
        bon_attn = [None, self.node_attn_to_edge_attn(ver_attn,edge_attn, e_index[0]), self.node_attn_to_edge_attn(edge_attn, cell_attn, c_index[0])]
        up_attn = [self.node_attn_to_edge_attn(ver_attn,ver_attn, v_index[1]), self.node_attn_to_edge_attn(edge_attn,edge_attn, e_index[1]), None]
        down_attn = [None, self.node_attn_to_edge_attn(edge_attn,edge_attn, e_index[2]), self.node_attn_to_edge_attn(cell_attn,cell_attn, c_index[2])]
        
        masked_clf_logits = self.clf(data, up_attn = up_attn, down_attn = down_attn, bon_attn = bon_attn, ph_x = ph_x)

        loss, loss_dict = self.__loss__(cell_attn_log_logits[0].sigmoid(), cell_attn_log_logits[1].sigmoid(), cell_attn_log_logits[2].sigmoid(), masked_clf_logits, data.y, epoch, warmup, tpl, gas)
        return loss, loss_dict, original_clf_logits, masked_clf_logits, ver_attn.reshape(-1), edge_attn.reshape(-1), cell_attn.reshape(-1)

    def get_r(self, current_epoch, final_r):
        r = self.init_r - current_epoch // self.decay_interval * self.decay_r
        if r < final_r:
            r = final_r
        return r

    def sampling(self, attn_log_logits, do_sampling):
        if do_sampling:
            random_noise = torch.empty_like(attn_log_logits).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            attn_bern = ((attn_log_logits + random_noise) / self.temperature).sigmoid()
        else:
            attn_bern = (attn_log_logits).sigmoid()
        return attn_bern

    @staticmethod
    def node_attn_to_edge_attn(src_attn, dst_attn, edge_index):
        if edge_index is None:
            return None
        src_attn = src_attn[edge_index[0]]
        dst_attn = dst_attn[edge_index[1]]
        edge_attn = src_attn * dst_attn
        return edge_attn
    
    @staticmethod
    def min_max_normalize(attn):
        attn_min = attn.min()
        attn_max = attn.max()
        epsilon = 1e-8
        attn_normalized = (attn - attn_min) / (attn_max - attn_min + epsilon)
        return attn_normalized
    
    def read_ph(self,data):
        emb, graph_x = self.clf.get_emb(data)
        cell_attn_log_logits = self.extractor(emb)
        ver_attn = self.sampling(cell_attn_log_logits[0], False)
        edge_attn = self.sampling(cell_attn_log_logits[1], False)
        cell_attn = self.sampling(cell_attn_log_logits[2], False)
        beta_0_up, beta_0_down, beta0_ext, beta1_ext, cyl= self.compute_extended_ph_link_tree_wcyclereps(ver_attn, data, self.device)
        print('beta0_ext',beta0_ext[0])
        # print('beta_0_down',beta_0_down[0])
        # print('beta_0_up',beta_0_up[0])
        print('beta1_ext',beta1_ext[0])
        print('cyl', cyl[0])