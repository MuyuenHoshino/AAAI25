import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer with edge features
    
"""
from layers.mlp_readout_layer import MLPReadout
from layers.SGGN_layer import SGGN_layer


class SGGNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()

 
    def __init__(self, net_params):
        super().__init__()

        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']

        hidden_dim = net_params['hidden_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        n_layers = net_params['L']
        d_conv = net_params['d_Conv']
        expand = net_params['hp_Expand']
        max_degree = net_params['max_degree']
        linear_agg = net_params['linear_agg']

        max_wl_role_index = 100 
        
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        
        self.embedding_h = nn.Linear(32, hidden_dim) # node feat is an integer
        self.embedding_e = nn.Embedding(2, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([SGGN_layer(hidden_dim, d_conv, expand, max_degree, linear_agg, update_E=False) for _ in range(n_layers) ]) 
        
        self.MLP_layer = MLPReadout(3 * hidden_dim, 1)


    def forward(self, g, h, e,  h_lap_pos_enc=None, h_wl_pos_enc=None):

        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc

        e = self.embedding_e(e)   

        
        g.ndata["hs"] = h
        g.edata["hs_e"] = e

        degrees = g.in_degrees().float().view(-1, 1)  # 获取入度，并转换为浮点数形式
        g.ndata['degree'] = degrees

        for conv in self.layers:
            g = conv(g)  

        hg_sum = dgl.sum_nodes(g, 'hs')
        hg_max = dgl.max_nodes(g, 'hs')
        hg_mean = dgl.mean_nodes(g, 'hs')
    
        hg = torch.cat((hg_sum, hg_max, hg_mean),dim = 1)
            
        return self.MLP_layer(hg)
        
    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss


