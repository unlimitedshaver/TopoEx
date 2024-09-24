import torch
import torch.nn.functional as F
from torch.nn import Linear
from typing import Any, Callable, Optional
from torch import Tensor
from torch_geometric.nn.inits import reset
from backbones.cell_mp import CochainMessagePassing, CochainMessagePassingParams

from torch.nn import Linear, Embedding, Sequential, BatchNorm1d as BN
from backbones.layers import InitReduceConv, EmbedVEWithReduce, OGBEmbedVEWithReduce
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from utils.cell_complex import ComplexBatch
from backbones.nn import pool_complex, get_pooling_fn, get_nonlinearity, get_graph_norm

import copy


class OGBEmbedCINpp(torch.nn.Module):
    def __init__(self, x_dim, edge_attr_dim, out_size, num_layers, hidden, dropout_rate: float = 0.5, 
                 indropout_rate: float = 0.0, max_dim: int = 2, jump_mode=None,
                 nonlinearity='relu', readout='sum', train_eps=False, final_hidden_multiplier: int = 2,
                 readout_dims=(0, 1, 2), final_readout='sum', apply_dropout_before='lin2',
                 init_reduce='sum', embed_edge=False, embed_dim=None, use_coboundaries=False,
                 graph_norm='bn', atom_encoder=False):
        super(OGBEmbedCINpp, self).__init__()

        self.max_dim = max_dim
        if readout_dims is not None:
            self.readout_dims = tuple([dim for dim in readout_dims if dim <= max_dim])
        else:
            self.readout_dims = list(range(max_dim+1))

        if embed_dim is None:
            embed_dim = hidden
        
        if atom_encoder is True:
            self.v_embed_init = AtomEncoder(embed_dim)
            self.e_embed_init = None
            if embed_edge:
                self.e_embed_init = BondEncoder(embed_dim)
        else:
            self.v_embed_init = Linear(x_dim, embed_dim)
            self.e_embed_init = None
            if embed_edge:
                self.e_embed_init = Linear(edge_attr_dim, embed_dim)

        self.reduce_init = InitReduceConv(reduce=init_reduce)

        self.init_conv = OGBEmbedVEWithReduce(self.v_embed_init, self.e_embed_init, self.reduce_init)

        self.final_readout = final_readout
        self.dropout_rate = dropout_rate
        self.in_dropout_rate = indropout_rate
        self.apply_dropout_before = apply_dropout_before
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.readout = readout
        act_module = get_nonlinearity(nonlinearity, return_module=True)
        self.graph_norm = get_graph_norm(graph_norm)
        for i in range(num_layers):
            layer_dim = embed_dim if i == 0 else hidden
            self.convs.append(
                CINppConv(up_msg_size=layer_dim, down_msg_size=layer_dim,
                    boundary_msg_size=layer_dim, passed_msg_boundaries_nn=None,
                    passed_msg_up_nn=None, passed_msg_down_nn=None, passed_update_up_nn=None,
                    passed_update_down_nn=None, passed_update_boundaries_nn=None, train_eps=train_eps,
                    max_dim=self.max_dim, hidden=hidden, act_module=act_module, layer_dim=layer_dim,
                    graph_norm=self.graph_norm, use_coboundaries=use_coboundaries))
        self.lin1s = torch.nn.ModuleList()
        for _ in range(max_dim + 1):  
            self.lin1s.append(Linear(hidden, final_hidden_multiplier * hidden))
        self.lin2 = Linear(final_hidden_multiplier * hidden, out_size)

        self.lin_ph = Linear(final_hidden_multiplier * hidden + 64, final_hidden_multiplier * hidden) ## 64 = num_structure_elements * 2 *2

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.init_conv.reset_parameters()
        self.lin1s.reset_parameters()
        self.lin2.reset_parameters()
       
    def forward(self, data: ComplexBatch,up_attn = [None,None,None], down_attn = [None,None,None], bon_attn = [None,None,None], ph_x = None):
        act = get_nonlinearity(self.nonlinearity, return_module=False)
        xs = None
        # res = {}
        # Embed and populate higher-levels 【populate：填充】
        params = data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=True)
        xs = list(self.init_conv(*params))

        # Apply dropout on the input features
        for i, x in enumerate(xs):
            xs[i] = F.dropout(xs[i], p=self.in_dropout_rate, training=self.training)

        emb_data = copy.deepcopy(data)
        emb_data.set_xs(xs)

        for c, conv in enumerate(self.convs):
            params = emb_data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=True)
            start_to_process = 0
            xs = conv(*params, start_to_process=start_to_process, up_attn = up_attn, down_attn = down_attn, bon_attn = bon_attn)
            # Apply dropout on the output of the conv layer
            for i, x in enumerate(xs):
                xs[i] = F.dropout(xs[i], p=self.dropout_rate, training=self.training)
            emb_data.set_xs(xs)

            # if include_partial:
            #     for k in range(len(xs)):
            #         res[f"layer{c}_{k}"] = xs[k]

        xs = pool_complex(xs, emb_data, self.max_dim, self.readout)  
        # Select the dimensions we want at the end.
        xs = [xs[i] for i in self.readout_dims]

        # if include_partial:
        #     for k in range(len(xs)):
        #         res[f"pool_{k}"] = xs[k]
        
        new_xs = []
        for i, x in enumerate(xs):
            if self.apply_dropout_before == 'lin1':
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            new_xs.append(act(self.lin1s[self.readout_dims[i]](x)))

        x = torch.stack(new_xs, dim=0)
        
        if self.apply_dropout_before == 'final_readout':
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        if self.final_readout == 'mean':
            x = x.mean(0)
        elif self.final_readout == 'sum':       ###
            x = x.sum(0)
        else:
            raise NotImplementedError

        ## cat ph_x
        if ph_x is not None:
            x = torch.cat((ph_x, x), dim=1)
            x = self.lin_ph(x)

        if self.apply_dropout_before not in ['lin1', 'final_readout']:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.lin2(x)

        # if include_partial:
        #     res['out'] = x
        #     return x, res
        return x
    
    def __repr__(self):
        return self.__class__.__name__
    
    def get_emb(self, data: ComplexBatch):    ##xs断在global_add_pool之前
        act = get_nonlinearity(self.nonlinearity, return_module=False)
        xs = None

        # Embed and populate higher-levels 【populate：填充】
        params = data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=True)
        xs = list(self.init_conv(*params))

        # # Apply dropout on the input features
        for i, x in enumerate(xs):
            xs[i] = F.dropout(xs[i], p=self.in_dropout_rate, training=self.training)

        emb_data = copy.deepcopy(data)     ##不要改变data
        emb_data.set_xs(xs)

        for c, conv in enumerate(self.convs):
            params = emb_data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=True)
            start_to_process = 0
            xs = conv(*params, start_to_process=start_to_process)
            # Apply dropout on the output of the conv layer
            for i, x in enumerate(xs):
                xs[i] = F.dropout(xs[i], p=self.dropout_rate, training=self.training)
            emb_data.set_xs(xs)

        cell_xs = xs
        xs = pool_complex(xs, emb_data, self.max_dim, self.readout)
        # Select the dimensions we want at the end.
        xs = [xs[i] for i in self.readout_dims]

        new_xs = []
        for i, x in enumerate(xs):
            if self.apply_dropout_before == 'lin1':
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            new_xs.append(act(self.lin1s[self.readout_dims[i]](x)))

        x = torch.stack(new_xs, dim=0)
        
        if self.apply_dropout_before == 'final_readout':
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        if self.final_readout == 'mean':
            x = x.mean(0)
        elif self.final_readout == 'sum':       ###
            x = x.sum(0)
        else:
            raise NotImplementedError
        
        if self.apply_dropout_before not in ['lin1', 'final_readout']:
            graph_x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        return cell_xs, graph_x
    

class CINppCochainConv(CochainMessagePassing):
    """This is a CIN Cochain layer that operates of boundaries and upper adjacent cells."""
    def __init__(self, dim: int,
                 up_msg_size: int,
                 down_msg_size: int,
                 boundary_msg_size: Optional[int],
                 msg_up_nn: Callable,                   ##up,boun,down
                 msg_boundaries_nn: Callable,
                 msg_down_nn: Callable,
                 update_up_nn: Callable,                ##update:u,b,d
                 update_boundaries_nn: Callable,
                 update_down_nn: Callable,
                 combine_nn: Callable,
                 eps: float = 0.,
                 train_eps: bool = False):
        super(CINppCochainConv, self).__init__(up_msg_size, down_msg_size, boundary_msg_size=boundary_msg_size,
                                                 use_down_msg=True)
        self.dim = dim
        self.msg_up_nn = msg_up_nn
        self.msg_boundaries_nn = msg_boundaries_nn
        self.msg_down_nn = msg_down_nn
        self.update_up_nn = update_up_nn
        self.update_boundaries_nn = update_boundaries_nn
        self.update_down_nn = update_down_nn
        self.combine_nn = combine_nn
        self.initial_eps = eps
        if train_eps:
            self.eps1 = torch.nn.Parameter(torch.Tensor([eps]))
            self.eps2 = torch.nn.Parameter(torch.Tensor([eps]))
            self.eps3 = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps1', torch.Tensor([eps]))
            self.register_buffer('eps2', torch.Tensor([eps]))
            self.register_buffer('eps3', torch.Tensor([eps]))
        self.reset_parameters()

    def forward(self, cochain: CochainMessagePassingParams, up_attn = None, down_attn = None, bon_attn = None):
        out_up, out_down, out_boundaries = self.propagate(cochain.up_index, cochain.down_index,
                                              cochain.boundary_index, x=cochain.x,
                                              up_attr=cochain.kwargs['up_attr'],
                                              down_attr=cochain.kwargs['down_attr'],  ###原代码缺少
                                              boundary_attr=cochain.kwargs['boundary_attr'],
                                              up_attn = up_attn, down_attn = down_attn, bon_attn = bon_attn)

        # As in GIN, we can learn an injective update function for each multi-set
        out_up += (1 + self.eps1) * cochain.x
        out_down += (1 + self.eps2) * cochain.x
        out_boundaries += (1 + self.eps3) * cochain.x
        out_up = self.update_up_nn(out_up)
        out_down = self.update_down_nn(out_down)
        out_boundaries = self.update_boundaries_nn(out_boundaries)

        # We need to combine the three such that the output is injective
        # Because the cross product of countable spaces is countable, then such a function exists.
        # And we can learn it with another MLP.
        return self.combine_nn(torch.cat([out_up, out_down, out_boundaries], dim=-1))

    def reset_parameters(self):
        reset(self.msg_up_nn)
        reset(self.msg_boundaries_nn)
        reset(self.update_up_nn)
        reset(self.update_boundaries_nn)
        reset(self.msg_down_nn)
        reset(self.update_down_nn)
        reset(self.combine_nn)
        self.eps1.data.fill_(self.initial_eps)
        self.eps2.data.fill_(self.initial_eps)
        self.eps3.data.fill_(self.initial_eps)

    def message_up(self, up_x_j: Tensor, up_attr: Tensor, up_attn = None) -> Tensor:
        msg = self.msg_up_nn((up_x_j, up_attr))
        if up_attn is not None:
            msg = msg * up_attn
        return msg
    
    def message_boundary(self, boundary_x_j: Tensor, bon_attn = None) -> Tensor:
        msg = self.msg_boundaries_nn(boundary_x_j)
        if bon_attn is not None:
            msg = msg * bon_attn
        return msg
    
    def message_down(self, down_x_j: Tensor, down_attr: Tensor, down_attn = None) -> Tensor:
        msg = self.msg_down_nn((down_x_j, down_attr))
        if down_attn is not None:
            msg = msg * down_attn
        return msg


class Catter(torch.nn.Module):
    def __init__(self):
        super(Catter, self).__init__()

    def forward(self, x):
        return torch.cat(x, dim=-1)
    

class CINppConv(torch.nn.Module):
    """A cellular version of GIN which performs message passing from  cellular upper
    neighbors and boundaries and lower neighbors (hence why "Sparse")
    """

    # TODO: Refactor the way we pass networks externally to allow for different networks per dim.
    def __init__(self, up_msg_size: int, down_msg_size: int, boundary_msg_size: Optional[int],
                 passed_msg_up_nn: Optional[Callable], passed_msg_boundaries_nn: Optional[Callable],passed_msg_down_nn: Optional[Callable],
                 passed_update_up_nn: Optional[Callable],
                 passed_update_boundaries_nn: Optional[Callable],passed_update_down_nn: Optional[Callable],
                 eps: float = 0., train_eps: bool = False, max_dim: int = 2,
                 graph_norm=BN, use_coboundaries=False, **kwargs):
        super(CINppConv, self).__init__()
        self.max_dim = max_dim
        self.mp_levels = torch.nn.ModuleList()
        for dim in range(max_dim+1):
            msg_up_nn = passed_msg_up_nn
            if msg_up_nn is None:
                if use_coboundaries:
                    msg_up_nn = Sequential(
                            Catter(),
                            Linear(kwargs['layer_dim'] * 2, kwargs['layer_dim']),
                            kwargs['act_module']())
                else:
                    msg_up_nn = lambda xs: xs[0]

            msg_down_nn = passed_msg_down_nn
            if msg_down_nn is None:
                if use_coboundaries:
                    msg_down_nn = Sequential(
                            Catter(),
                            Linear(kwargs['layer_dim'] * 2, kwargs['layer_dim']),
                            kwargs['act_module']())
                else:
                    msg_down_nn = lambda xs: xs[0]

            msg_boundaries_nn = passed_msg_boundaries_nn
            if msg_boundaries_nn is None:
                msg_boundaries_nn = lambda x: x

            update_up_nn = passed_update_up_nn
            if update_up_nn is None:
                update_up_nn = Sequential(
                    Linear(kwargs['layer_dim'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module'](),
                    Linear(kwargs['hidden'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module']()
                )

            update_down_nn = passed_update_down_nn
            if update_down_nn is None:
                update_down_nn = Sequential(
                    Linear(kwargs['layer_dim'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module'](),
                    Linear(kwargs['hidden'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module']()
                )

            update_boundaries_nn = passed_update_boundaries_nn
            if update_boundaries_nn is None:
                update_boundaries_nn = Sequential(
                    Linear(kwargs['layer_dim'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module'](),
                    Linear(kwargs['hidden'], kwargs['hidden']),
                    graph_norm(kwargs['hidden']),
                    kwargs['act_module']()
                )
            combine_nn = Sequential(
                Linear(kwargs['hidden']*3, kwargs['hidden']),
                graph_norm(kwargs['hidden']),
                kwargs['act_module']())

            mp = CINppCochainConv(dim, up_msg_size, down_msg_size, boundary_msg_size=boundary_msg_size,
                msg_up_nn=msg_up_nn, msg_down_nn=msg_down_nn, msg_boundaries_nn=msg_boundaries_nn, update_up_nn=update_up_nn,
                update_down_nn=update_down_nn, update_boundaries_nn=update_boundaries_nn, combine_nn=combine_nn, eps=eps,
                train_eps=train_eps)
            self.mp_levels.append(mp)

    def forward(self, *cochain_params: CochainMessagePassingParams, start_to_process=0, up_attn = [None, None, None], down_attn = [None, None, None], bon_attn = [None, None, None]):
        assert len(cochain_params) <= self.max_dim+1

        out = []
        for dim in range(len(cochain_params)):
            if dim < start_to_process:
                out.append(cochain_params[dim].x)
            else:
                out.append(self.mp_levels[dim].forward(cochain_params[dim],up_attn = up_attn[dim], down_attn = down_attn[dim], bon_attn = bon_attn[dim]))
        return out


