import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer with edge features
    
"""

from layers.mlp_readout_layer import MLPReadout
from layers.SGGN_layer import SGGN_layer, SGGN_layer_withoutC

from torch.profiler import profile, record_function, ProfilerActivity



class SGGNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()

 
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']

        hidden_dim = net_params['hidden_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        n_layers = net_params['L']
        d_conv = net_params['d_Conv']
        expand = net_params['hp_Expand']
        max_degree = net_params['max_degree']
        linear_agg = net_params['linear_agg']

        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        max_wl_role_index = 37 # this is maximum graph size in the dataset
        
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        

        self.embedding_h = nn.Embedding(num_atom_type + 1, hidden_dim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        
        self.layers = nn.ModuleList([SGGN_layer(hidden_dim, d_conv, expand, max_degree, linear_agg, update_E=True) for _ in range(n_layers) ]) 

        self.MLP_layer = MLPReadout(3 * hidden_dim, 1)   # 1 out dim since regression problem   




        
    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):

        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc
        if not self.edge_feat: # edge feature set to 1
            e = torch.ones(e.size(0),1).to(self.device)
        
        e = self.embedding_e(e) 

        
        g.ndata["hs"] = h
        g.edata["hs_e"] = e
        
        degrees = g.in_degrees().float().view(-1, 1) 
        g.ndata['degree'] = degrees

        for conv in self.layers:
            g = conv(g)     


        g.ndata['h'] = g.ndata['hs']


        self.readout = "3 readout"


        hg_sum = dgl.sum_nodes(g, 'h')
        hg_max = dgl.max_nodes(g, 'h')
        hg_mean = dgl.mean_nodes(g, 'h')
    
        hg = torch.cat((hg_sum, hg_max, hg_mean),dim = 1)
            
        return self.MLP_layer(hg)
        
    def loss(self, scores, targets):
        loss = nn.L1Loss()(scores, targets)
        return loss








class SGGNNet_withoutC(nn.Module):
    def __init__(self, net_params):
        super().__init__()

 
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']

        hidden_dim = net_params['hidden_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        n_layers = net_params['L']
        d_conv = net_params['d_Conv']
        expand = net_params['hp_Expand']
        max_degree = net_params['max_degree']
        linear_agg = net_params['linear_agg']

        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.lap_pos_enc = net_params['lap_pos_enc']
        self.wl_pos_enc = net_params['wl_pos_enc']
        max_wl_role_index = 37 # this is maximum graph size in the dataset
        
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)
        

        self.embedding_h = nn.Embedding(num_atom_type + 1, hidden_dim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        
        self.layers = nn.ModuleList([SGGN_layer_withoutC(hidden_dim, d_conv, expand, max_degree, linear_agg, update_E=True) for _ in range(n_layers) ]) 

        self.MLP_layer = MLPReadout(3 * hidden_dim, 1)   # 1 out dim since regression problem   




        
    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):

        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc) 
            h = h + h_wl_pos_enc
        if not self.edge_feat: # edge feature set to 1
            e = torch.ones(e.size(0),1).to(self.device)
        
        e = self.embedding_e(e) 

        
        g.ndata["hs"] = h
        g.edata["hs_e"] = e
        
        degrees = g.in_degrees().float().view(-1, 1) 
        g.ndata['degree'] = degrees

        for conv in self.layers:
            g = conv(g)     


        g.ndata['h'] = g.ndata['hs']


        self.readout = "3 readout"


        hg_sum = dgl.sum_nodes(g, 'h')
        hg_max = dgl.max_nodes(g, 'h')
        hg_mean = dgl.mean_nodes(g, 'h')
    
        hg = torch.cat((hg_sum, hg_max, hg_mean),dim = 1)
            
        return self.MLP_layer(hg)
        
    def loss(self, scores, targets):
        loss = nn.L1Loss()(scores, targets)
        return loss