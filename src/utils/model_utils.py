import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm
from torch.nn.parameter import Parameter
import torch.nn.functional as F

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


def get_optimizer(clf, extractor, optimizer_config, method_config, warmup, slayer, gaus):
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
        return algo([{'params': extractor.parameters(), 'lr': attn_lr, 'weight_decay': attn_wd}, {'params': clf_base_params},
                     {'params':slayer.parameters(),'lr': emb_lr, 'weight_decay': emb_wd}, {'params':gaus.parameters(),'lr': 0.001},
                     {'params': clf_emb_model_params, 'lr': emb_lr, 'weight_decay': emb_wd}], lr=pred_lr, weight_decay=pred_wd)
    

class PhAttn(nn.Module):
    def __init__(self, hidden_size, num_heads=2, input_dim=4):
        super(PhAttn, self).__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # define mutihead linear
        self.ph_emb = nn.Linear(input_dim, hidden_size, bias=False)
        self.graph_emb = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, graph_x, ph_x):
        """
        param:
            graph_x: [batch_size, hidden_size]
            ph_x: [batch_size, num_structure, input_dim]
        
        return:
            output: [batch_size, num_structure, input_dim]
        """
        batch_size, num_structure, input_dim = ph_x.size()
        # muti-head Linear projection
        ph_x_proj = self.ph_emb(ph_x)  # [batch_size, num_structure, hidden_size]
        graph_x_proj = self.graph_emb(graph_x)  # [batch_size, hidden_size]
        # reshape to muti-head format
        ph_x_proj = ph_x_proj.view(batch_size, num_structure, self.num_heads, self.head_dim) # [batch_size, num_structure, num_heads, head_dim]
        graph_x_proj = graph_x_proj.view(batch_size, self.num_heads, self.head_dim) # [batch_size, num_heads, head_dim]
        # permute dimensions for matrix operations
        ph_x_proj = ph_x_proj.permute(0, 2, 1, 3) # [batch_size, num_heads, num_structure, head_dim]
        graph_x_proj = graph_x_proj.unsqueeze(-1) # [batch_size, num_heads, head_dim, 1]
        # calculate attention score: head_dim*head_dim
        attn_logits = torch.matmul(ph_x_proj, graph_x_proj).squeeze(-1) # [batch_size, num_heads, num_structure]
        # softmax
        attn = F.softmax(attn_logits, dim=-1)  # [batch_size, num_heads, num_structure]
        # weighted sum, broadcast on num_heads
        attn = attn.unsqueeze(-1) # [batch_size, num_heads, num_structure, 1]
        weighted_x = attn * ph_x.unsqueeze(1)  # [batch_size, num_heads, num_structure, input_dim]
        weighted_x = weighted_x.view(batch_size, self.num_heads, num_structure, input_dim) # [batch_size, num_heads, num_structure, input_dim]
        # combine head
        output = weighted_x.mean(dim=1) 
        return output

    # def orthogonal_loss(self, attn1, attn2):
    #     # Compute orthogonal loss
    #     dot_product = torch.sum(attn1 * attn2, dim=1)
    #     orthogonal_loss = (dot_product ** 2).mean()
    #     return orthogonal_loss



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
        self.ldgm_h2 = SLayerRationalHat(num_struct_elements, 2,
                                         radius_init=0.1)
        self.num_struct_elements = num_struct_elements
        
        self.ph_attn_1 = PhAttn(128) # 2*graph_x.shape[1]

        self.ph_attn_2 = PhAttn(128) # 2*graph_x.shape[1]
        
    def do_ph_attn(self, ph, ldgm, graph_x, ph_attn):
        ph_out, centers, radius = ldgm(ph) # [bs,num_struct], [num_struct,2], [num_struct]
        ph_out = ph_out.unsqueeze(-1)
        centers = centers.unsqueeze(0).expand(ph_out.size(0),self.num_struct_elements,2)
        radius = radius.unsqueeze(0).unsqueeze(-1).expand(ph_out.size(0),self.num_struct_elements,1)
        to_attn = torch.cat((ph_out,centers, radius), dim=-1)
        return ph_attn(graph_x, to_attn)

    def forward(self, beta_0_up, beta_0_down, beta0_ext, beta1_ext, graph_x, cyl):
        
        input_device = beta_0_up[0].device
        # transform coordinates
        # x_0_ex = [beta[:,0] for beta in beta0_ext]
        # y_0_ex = [beta[:,1] for beta in beta0_ext]
        # beta0_ext_1sthalf = [torch.stack([x_0, 1-x_0], dim=1) for x_0 in x_0_ex]
        # beta0_ext_2sthalf = [torch.stack([y_0, 1-y_0], dim=1) for y_0 in y_0_ex]

        x_1_ex = [beta[:,0] for beta in beta1_ext]
        y_1_ex = [beta[:,1] for beta in beta1_ext]
        beta1_ext_1sthalf = [torch.stack([x_1, torch.ones_like(x_1)], dim=1) for x_1 in x_1_ex]
        beta1_ext_2sthalf = [torch.stack([1.0-y_1, torch.ones_like(y_1)], dim=1) for y_1 in y_1_ex]

        # define beta2 => ([x_1, 1+cyl_len], [y_1, 1+cyl_len])
        cyl_len = [torch.stack([torch.tensor(c.size(0)+1, device=input_device) for c in g]) for g in cyl]
        ## norm
        cyl_len_tensor = torch.cat(cyl_len)
        max_value = torch.max(cyl_len_tensor)
        min_value = torch.min(cyl_len_tensor)
        norm_cyl_len = []
        for g in cyl_len:
            normalized_g = (g - min_value + 1e-6) / (max_value - min_value + 1e-5)
            normalized_g = normalized_g + 1
            norm_cyl_len.append(normalized_g)
        ## combine
        beta2_1sthalf = []
        for idx in range(len(x_1_ex)):
            b21 = torch.stack([x_1_ex[idx], norm_cyl_len[idx]], dim=1)
            beta2_1sthalf.append(b21)
        beta2_2sthalf = []
        for idx in range(len(x_1_ex)):
            b22 = torch.stack([1-y_1_ex[idx], norm_cyl_len[idx]], dim=1)
            beta2_2sthalf.append(b22)

        # combine
        # ph_down = beta_0_down || beta0_ext_2sthalf || beta1_ext_2sthalf || beta2_2sthalf
        # ph_up = beta_0_up || beta0_ext_1sthalf || beta1_ext_1sthalf || beta2_1sthalf
        ph_up = [torch.cat((beta1_ext_1sthalf[i], beta2_1sthalf[i]), dim=0) for i in range(len(beta_0_up))]
        ph_down = [torch.cat((beta1_ext_2sthalf[i], beta2_2sthalf[i]), dim=0) for i in range(len(beta_0_up))]

        # ph_up_out, up_centers, up_r = self.ldgm_h1(ph_up) # [bs,num_struct], [num_struct,2], [num_struct]
        # ph_up_out = ph_up_out.unsqueeze(-1)
        # up_centers = up_centers.unsqueeze(0).expand(ph_up_out.size(0),self.num_struct_elements,2)
        # up_r = up_r.unsqueeze(0).unsqueeze(-1).expand(ph_up_out.size(0),self.num_struct_elements,1)
        # up_to_attn = torch.cat((ph_up_out,up_centers,up_r), dim=-1)
        # ph_up_out = self.ph_attn(graph_x,up_to_attn)

        ph_up_out_1 = self.do_ph_attn(ph_up, self.ldgm_h1, graph_x, self.ph_attn_1)
    
        # ph_down_out, down_centers, down_r = self.ldgm_h1(ph_down)
        # ph_down_out = ph_down_out.unsqueeze(-1)
        # down_centers = down_centers.unsqueeze(0).expand(ph_down_out.size(0),self.num_struct_elements,2)
        # down_r = down_r.unsqueeze(0).unsqueeze(-1).expand(ph_down_out.size(0),self.num_struct_elements,1)
        # down_to_attn = torch.cat((ph_down_out, down_centers, down_r), dim=-1)
        # ph_down_out = self.ph_attn(graph_x,down_to_attn)

        ph_down_out_1 = self.do_ph_attn(ph_down, self.ldgm_h1, graph_x, self.ph_attn_1)

        ph_up_out_2 = self.do_ph_attn(beta_0_up, self.ldgm_h2, graph_x, self.ph_attn_2)

        ph_down_out_2 = self.do_ph_attn(beta_0_down, self.ldgm_h2, graph_x, self.ph_attn_2)

        # MSE
        tpl = -((ph_up_out_1 - ph_down_out_1)**2).sum() - ((ph_up_out_2 - ph_down_out_2)**2).sum()

        ph_up_out_1 = ph_up_out_1[:,:,0]
        ph_down_out_1 = ph_down_out_1[:,:,0]
        ph_up_out_2 = ph_up_out_2[:,:,0]
        ph_down_out_2 = ph_down_out_2[:,:,0]

        x = torch.cat([ph_up_out_1, ph_down_out_1, ph_up_out_2, ph_down_out_2], dim=1)
        return x,tpl
    

    
class GaussianMixtureModel(nn.Module):
    def __init__(self, mu1=0.75, mu2=0.25, variance_penalty=1e-3):
        super().__init__()
        self.mu1 = mu1
        self.mu2 = mu2

        self.s1 = nn.Parameter(torch.log(torch.tensor(0.25)))  # r1 = 0.25
        self.s2 = nn.Parameter(torch.log(torch.tensor(0.25)))  # r2 = 0.25

        self.b = nn.Parameter(torch.tensor(0.0))  # a = sigmoid(0) = 0.5

        self.variance_penalty = variance_penalty

    def forward(self, x):

        r1 = torch.exp(self.s1)  
        r2 = torch.exp(self.s2)  
        a = 0.5  # torch.sigmoid(self.b)

        # PDF
        N1 = (1 / torch.sqrt(2 * torch.pi * r1)) * torch.exp(-0.5 * ((x - self.mu1) ** 2) / r1)
        N2 = (1 / torch.sqrt(2 * torch.pi * r2)) * torch.exp(-0.5 * ((x - self.mu2) ** 2) / r2)

        # mix PDF
        p = a * N1 + (1 - a) * N2

        nll = -torch.sum(torch.log(p + 1e-10), dim=1)  

        variance_control = 1.0 / (r1 + 1e-6) + 1.0 / (r2 + 1e-6) 

        loss = torch.mean(nll) + self.variance_penalty * torch.mean(variance_control)

        return loss

