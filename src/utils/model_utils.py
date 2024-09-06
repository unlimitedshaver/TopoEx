import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm

from torchPHext.torchex_PHext.nn import SLayerRationalHat, SLayerSquare, SLayerExponential


class MLP(nn.Sequential):
    def __init__(self, channels, dropout_p, norm_type, act_type):
        norm = self.get_norm(norm_type)
        act = self.get_act(act_type)

        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i]))

            if i < len(channels) - 1:
                m.append(norm(channels[i]))
                m.append(act())
                m.append(nn.Dropout(dropout_p))

        super(MLP, self).__init__(*m)

    @staticmethod
    def get_norm(norm_type):
        if isinstance(norm_type, str) and 'batch' in norm_type:
            return BatchNorm
        elif norm_type == 'none' or norm_type is None:
            return nn.Identity
        else:
            raise ValueError('Invalid normalization type: {}'.format(norm_type))

    @staticmethod
    def get_act(act_type):
        if act_type == 'relu':
            return nn.ReLU
        elif act_type == 'silu':
            return nn.SiLU
        elif act_type == 'none':
            return nn.Identity
        else:
            raise ValueError('Invalid activation type: {}'.format(act_type))


class CoorsNorm(nn.Module):
    def __init__(self, eps = 1e-6, scale_init = 1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim = -1, keepdim = True)
        normed_coors = coors / norm.clamp(min = self.eps)
        return normed_coors #* self.scale


class ExtractorMLP(nn.Module):

    def __init__(self, hidden_size, config, out_dim=None):
        super().__init__()

        dropout_p = config['dropout_p']
        norm_type = config['norm_type']
        act_type = config['act_type']
        # covar_dim = config.get('covar_dim', None)
        # if covar_dim is None:
        out_dim = 1
        attn_dim = out_dim
        self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, attn_dim], dropout_p, norm_type, act_type)

    def forward(self, emb):
        attn_log_logits = [self.feature_extractor(x) for x in emb]
        # attn_log_logits = [self.feature_extractor_0(emb[0]), self.feature_extractor_1(emb[1]), self.feature_extractor_2(emb[2])]
        return attn_log_logits


class FeatEncoder(torch.nn.Module):

    def __init__(self, hidden_size, categorical_feat, scalar_feat, n_categorical_feat_to_use=-1, n_scalar_feat_to_use=-1):
        super().__init__()
        self.embedding_list = torch.nn.ModuleList()

        self.num_categorical_feat = len(categorical_feat)
        self.n_categorical_feat_to_use = self.num_categorical_feat if n_categorical_feat_to_use == -1 else n_categorical_feat_to_use
        self.num_scalar_feat_to_use = scalar_feat if n_scalar_feat_to_use == -1 else n_scalar_feat_to_use

        for i in range(self.n_categorical_feat_to_use):
            num_categories = categorical_feat[i]
            emb = torch.nn.Embedding(num_categories, hidden_size)
            self.embedding_list.append(emb)

        if self.num_scalar_feat_to_use > 0:
            assert n_scalar_feat_to_use == -1
            self.linear = torch.nn.Linear(self.num_scalar_feat_to_use, hidden_size)

        total_cate_dim = self.n_categorical_feat_to_use*hidden_size
        self.dim_mapping = torch.nn.Linear(total_cate_dim + hidden_size, hidden_size) if self.num_scalar_feat_to_use > 0 else torch.nn.Linear(total_cate_dim, hidden_size)

    def forward(self, x):
        x_embedding = []
        for i in range(self.n_categorical_feat_to_use):
            x_embedding.append(self.embedding_list[i](x[:, i].long()))

        if self.num_scalar_feat_to_use > 0:
            x_embedding.append(self.linear(x[:, self.num_categorical_feat:]))

        x_embedding = self.dim_mapping(torch.cat(x_embedding, dim=-1))
        return x_embedding


def get_optimizer(clf, extractor, optimizer_config, method_config, warmup, slayer):
    pred_lr = method_config['pred_lr']
    pred_wd = method_config['pred_wd']

    wp_lr = optimizer_config['wp_lr']
    wp_wd = optimizer_config['wp_wd']
    attn_lr = optimizer_config['attn_lr']
    attn_wd = optimizer_config['attn_wd']
    emb_lr = optimizer_config['emb_lr']
    emb_wd = optimizer_config['emb_wd']

    algo = torch.optim.Adam
    clf_emb_model_params = [kv[1]  for kv in clf.named_parameters() if 'emb_model' in kv[0]]
    clf_base_params = [kv[1] for kv in clf.named_parameters() if 'emb_model' not in kv[0]]

    if warmup:
        return algo([{'params': clf_base_params}], lr=wp_lr, weight_decay=wp_wd)
    else:
        return algo([{'params': extractor.parameters(), 'lr': attn_lr, 'weight_decay': attn_wd}, {'params': clf_base_params}, {'params':slayer.parameters(),'lr': emb_lr, 'weight_decay': emb_wd},
                     {'params': clf_emb_model_params, 'lr': emb_lr, 'weight_decay': emb_wd}], lr=pred_lr, weight_decay=pred_wd)
    

class PershomReadout(nn.Module):
    def __init__(self,
                #  dataset,
                 num_struct_elements=None,
                #  cls_hidden_dimension=None,
                #  drop_out=None,
                 ):
        super().__init__()
        assert isinstance(num_struct_elements, int)

        # self.ldgm_0_up = SLayerRationalHat(num_struct_elements, 2, radius_init=0.1)
        # self.ldgm_0_down = SLayerRationalHat(num_struct_elements, 2, radius_init=0.1)
        # self.ldgm_cc = SLayerRationalHat(num_struct_elements, 2,
        #                                  radius_init=0.1)
        self.ldgm_h1 = SLayerRationalHat(num_struct_elements, 2,
                                         radius_init=0.1)


    def forward(self, beta_0_up, beta_0_down, beta0_ext, beta1_ext):
        
        # input_device = beta_0_up[0].device
        ## transform coordinates
        x_0_ex = [beta[:,0] for beta in beta0_ext]
        y_0_ex = [beta[:,1] for beta in beta0_ext]
        beta0_ext_1sthalf = [torch.stack([x_0, 1-x_0], dim=1) for x_0 in x_0_ex]
        beta0_ext_2sthalf = [torch.stack([y_0, 1-y_0], dim=1) for y_0 in y_0_ex]

        x_1_ex = [beta[:,0] for beta in beta1_ext]
        y_1_ex = [beta[:,1] for beta in beta1_ext]
        beta1_ext_1sthalf = [torch.stack([x_1, 1-x_1], dim=1) for x_1 in x_1_ex]
        beta1_ext_2sthalf = [torch.stack([y_1, 1-y_1], dim=1) for y_1 in y_1_ex]
        
        ## combine
        ##ph_down = beta_0_down || beta0_ext_1sthalf || beta1_ext_1sthalf
        ##ph_up = beta_0_up || beta0_ext_2sthalf || beta1_ext_2sthalf
        ph_down = [torch.cat((beta_0_down[i], beta0_ext_1sthalf[i], beta1_ext_1sthalf[i]), dim=0) for i in range(len(beta_0_up))]
        ph_up = [torch.cat((beta_0_up[i], beta0_ext_2sthalf[i], beta1_ext_2sthalf[i]), dim=0) for i in range(len(beta_0_up))]

        ph_up_out = self.ldgm_h1(ph_up)
        ph_down_out = self.ldgm_h1(ph_down)

        # MSE
        tpl = -((ph_up_out - ph_down_out)**2).sum()

        x = torch.cat([ph_up_out, ph_down_out], dim=1)
        return x,tpl
