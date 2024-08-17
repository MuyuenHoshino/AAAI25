import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from layers.mlp_readout_layer import MLPReadout
from layers.SGGN_layer import SGGN_layer, SGGN_layer_withoutC



class SGGNNet(nn.Module):
    
    def __init__(self, net_params):
    
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']

        hidden_dim = net_params['hidden_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        n_layers = net_params['L']
        d_conv = net_params['d_Conv']
        expand = net_params['hp_Expand']
        max_degree = net_params['max_degree']
        linear_agg = net_params['linear_agg']
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.embedding_e = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([SGGN_layer(hidden_dim, d_conv, expand, max_degree, linear_agg, update_E=False) for _ in range(n_layers) ]) 

        self.MLP_layer = MLPReadout(hidden_dim * 3, n_classes)
        
    def forward(self, g, h, e):

        h = self.embedding_h(h)
        e = self.embedding_e(e) 
        
        g.ndata["hs"] = h
        g.edata["hs_e"] = e

        degrees = g.in_degrees().float().view(-1, 1)
        g.ndata['degree'] = degrees

        for conv in self.layers:
            g = conv(g)
        g.ndata['h'] = g.ndata['hs']
        
        hg_sum = dgl.sum_nodes(g, 'h')
        hg_max = dgl.max_nodes(g, 'h')
        hg_mean = dgl.mean_nodes(g, 'h')
    
        hg = torch.cat((hg_sum, hg_max, hg_mean),dim = 1)
            
        return self.MLP_layer(hg)
        
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
    






class SGGNNet_withoutC(nn.Module):
    
    def __init__(self, net_params):
    
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']

        hidden_dim = net_params['hidden_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        n_layers = net_params['L']
        d_conv = net_params['d_Conv']
        expand = net_params['hp_Expand']
        max_degree = net_params['max_degree']
        linear_agg = net_params['linear_agg']
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
        self.embedding_e = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([SGGN_layer_withoutC(hidden_dim, d_conv, expand, max_degree, linear_agg, update_E=False) for _ in range(n_layers) ]) 

        self.MLP_layer = MLPReadout(hidden_dim * 3, n_classes)
        
    def forward(self, g, h, e):

        h = self.embedding_h(h)
        e = self.embedding_e(e) 
        
        g.ndata["hs"] = h
        g.edata["hs_e"] = e

        degrees = g.in_degrees().float().view(-1, 1)
        g.ndata['degree'] = degrees

        for conv in self.layers:
            g = conv(g)
        g.ndata['h'] = g.ndata['hs']
        
        hg_sum = dgl.sum_nodes(g, 'h')
        hg_max = dgl.max_nodes(g, 'h')
        hg_mean = dgl.mean_nodes(g, 'h')
    
        hg = torch.cat((hg_sum, hg_max, hg_mean),dim = 1)
            
        return self.MLP_layer(hg)
        
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss