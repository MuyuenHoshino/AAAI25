import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer
    
"""

from layers.mlp_readout_layer import MLPReadout
from layers.SGGN_layer import SGGN_layer_withoutE

from torch.profiler import profile, record_function, ProfilerActivity


class SGGNNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']

        in_dim_node = net_params['in_dim']
        n_classes = net_params['n_classes']
        self.n_classes = n_classes
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
        
        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim) # node feat is an integer
        self.embedding_e = nn.Linear(1, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)



        self.layers = nn.ModuleList([SGGN_layer_withoutE(hidden_dim, d_conv, expand, max_degree, linear_agg, update_E=False) for _ in range(n_layers) ]) 

        self.MLP_layer = MLPReadout(hidden_dim, n_classes)


    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):

        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc
        
        g.ndata["hs"] = h

        degrees = g.in_degrees().float().view(-1, 1)  # 获取入度，并转换为浮点数形式
        g.ndata['degree'] = degrees

        for conv in self.layers:
            g = conv(g)  
        h = g.ndata['hs']
            
        return self.MLP_layer(h)
    
    
    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss






        
