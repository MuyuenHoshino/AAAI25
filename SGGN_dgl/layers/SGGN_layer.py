import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
 








class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output



class GatedCNN(nn.Module):

    def __init__(self, d_model, d_conv, expand, **kwargs):
        super().__init__()

        hidden = d_model
        d_inner = hidden * expand

        self.fc1 = nn.Linear(hidden, hidden * expand * 2)
        self.act = F.silu
        self.fc2 = nn.Linear(hidden * expand, hidden)

        self.norm = RMSNorm(hidden)


        self.conv1d = nn.Conv1d(in_channels=d_inner, out_channels=d_inner, 
                              kernel_size=d_conv, bias=True, 
                              groups=d_inner,
                              padding=d_conv - 1)

    def forward(self, x):
        x_ori = x

        _, L, _ = x.shape
        xz = self.fc1(x) # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1) # (B, L, ED), (B, L, ED)

        x = x.transpose(1, 2) # (B, ED, L)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2) # (B, L, ED)
        
        x = self.act(x)
        z = self.act(z)
        
        output = x * z
        output = self.fc2(output) # (B, L, D)

        output = self.norm(output) + x_ori

        return output
    


class GatedCNN_withoutC(nn.Module):

    def __init__(self, d_model, d_conv, expand, **kwargs):
        super().__init__()

        hidden = d_model
        d_inner = hidden * expand

        self.fc1 = nn.Linear(hidden, hidden * expand * 2)
        self.act = F.silu
        self.fc2 = nn.Linear(hidden * expand, hidden)

        self.norm = RMSNorm(hidden)


    def forward(self, x):
        x_ori = x

        _, L, _ = x.shape
        xz = self.fc1(x) # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1) # (B, L, ED), (B, L, ED)

        
        x = self.act(x)
        z = self.act(z)
        
        output = x * z
        output = self.fc2(output) # (B, L, D)

        output = self.norm(output) + x_ori

        return output




class Updated_E_layer(nn.Module):
    def __init__(self, d_NodeAttr):
        super(Updated_E_layer, self).__init__()

        self.d_NodeAttr = d_NodeAttr
        self.act = nn.Sigmoid()

        self.combined_update_e = nn.Linear(d_NodeAttr, d_NodeAttr)


    def forward(self, g):
        g.update_all(message_func=self.message_func, reduce_func=self.reduce_func)

    def message_func(self, edges):

        edges.data['hs_e'] = self.act(self.combined_update_e(edges.data['hs_e'])) + edges.dst['hs'] + edges.src['hs'] + edges.data['hs_e']

        return {'msg': edges.dst['hs'] }

    def reduce_func(self, nodes):
        return {'useless': nodes.mailbox['msg'][:, 0, :]}
    

class Message_passing_layer(nn.Module):

    def __init__(self, max_degree, d_NodeAttr):
        super(Message_passing_layer, self).__init__()
        self.max_degree = max_degree
        self.d_NodeAttr = d_NodeAttr

        self.act = nn.Sigmoid()
        self.reduce = self.reduce_func


    def forward(self, g):
        
        g.edata['hs_e'] = self.act(g.edata['hs_e'])
        g.update_all(message_func=self.message_func, reduce_func=self.reduce)

    def message_func(self, edges):
        return {'msg': edges.dst['hs'] * edges.data["hs_e"], 
                'msg_sort_score': edges.dst['degree']}
    
    def reduce_func(self, nodes):
        _, n_neibor, D = nodes.mailbox['msg'].shape
        sort_score = nodes.mailbox['msg_sort_score'] + torch.rand(_, n_neibor, 1).cuda()
        msg = nodes.mailbox['msg']
        combined_tensor = torch.cat((msg, sort_score), dim=2)
        sorted_tensor, sorted_indices = torch.sort(combined_tensor[:, :, -1], dim=1)
        sorted_msg = torch.gather(msg, 1, sorted_indices.unsqueeze(-1).expand(-1, n_neibor, D))
        n_need_to_pad = self.max_degree - n_neibor
        padding_target_shape = (0, 0, 0, n_need_to_pad, 0, 0)
        return {'hs_aggregate': F.pad(sorted_msg, padding_target_shape, mode='constant', value=0)}
    


class Message_passing_layer_withoutE(nn.Module):

    def __init__(self, max_degree, d_NodeAttr):
        super(Message_passing_layer_withoutE, self).__init__()
        self.max_degree = max_degree
        self.d_NodeAttr = d_NodeAttr

        self.act = nn.Sigmoid()
        self.reduce = self.reduce_func

    def forward(self, g):
        

        g.update_all(message_func=self.message_func, reduce_func=self.reduce)

    def message_func(self, edges):
        return {'msg': edges.dst['hs'], 
                'msg_sort_score': edges.dst['degree']}
    
    def reduce_func(self, nodes):
        _, n_neibor, D = nodes.mailbox['msg'].shape
        sort_score = nodes.mailbox['msg_sort_score'] + torch.rand(_, n_neibor, 1).cuda()
        msg = nodes.mailbox['msg']
        combined_tensor = torch.cat((msg, sort_score), dim=2)
        sorted_tensor, sorted_indices = torch.sort(combined_tensor[:, :, -1], dim=1)
        sorted_msg = torch.gather(msg, 1, sorted_indices.unsqueeze(-1).expand(-1, n_neibor, D))
        n_need_to_pad = self.max_degree - n_neibor
        padding_target_shape = (0, 0, 0, n_need_to_pad, 0, 0)
        return {'hs_aggregate': F.pad(sorted_msg, padding_target_shape, mode='constant', value=0)}
    
    



class SGGN_layer(nn.Module):

    def __init__(self, d_hidden, d_Conv, hp_Expand, max_degree, linear_agg, update_E):
        super().__init__()
        self.d_hidden = d_hidden
        self.Group_Gating = GatedCNN(
        d_model=self.d_hidden, # Model dimension d_model
        d_conv=d_Conv,    # Local convolution width
        expand=hp_Expand,    # Block expansion factor
        ).to("cuda")

        self.MP = Message_passing_layer(max_degree, self.d_hidden)

        self.update_E = update_E
        if self.update_E:
            self.Upd_E = Updated_E_layer(d_hidden)
        
        self.BN_layer = nn.BatchNorm1d(self.d_hidden)

        self.relu = nn.ReLU6()

        self.linear_agg = linear_agg
        if linear_agg:
            self.linear_agg_layer = nn.Linear(max_degree+1, 1)


    def forward(self, g):
        
        hs_ori = g.ndata["hs"].float()
        hs = hs_ori.view(-1, 1, self.d_hidden)
        self.MP(g)     

        adj = g.adjacency_matrix()

        adj = torch.sparse_coo_tensor(
        adj.indices(),   
        adj.val,         
        size=adj.shape,  
        dtype=torch.float32).coalesce()
        n_nodes = adj.shape[0]
        eye_indices = torch.arange(n_nodes).unsqueeze(0).repeat(2,1).cuda()
        eye_values = torch.ones(n_nodes).cuda()
        new_indices = torch.cat([adj.indices(), eye_indices], dim=1)
        new_values = torch.cat([adj.values(), eye_values], dim = 0)
        adj = torch.sparse_coo_tensor(new_indices, new_values, (n_nodes, n_nodes)).coalesce().cuda()

        # print(adj_sparse)
        h = g.ndata["hs_aggregate"].float()

        h_input = torch.spmm(adj, h.view(n_nodes, -1))
        h_input = torch.cat((hs, h_input.view(n_nodes, -1, self.d_hidden)), dim = 1)


        if self.linear_agg:

            output = self.Group_Gating(h_input)
            output = output.transpose(1,2)
            output = self.BN_layer(output)

            output = self.relu(output)
            output = self.linear_agg_layer(output)
            output = output.transpose(1,2)
            output = output + hs
            
            g.ndata['hs'] = torch.squeeze(output)  


        else:

            output = self.Group_Gating(h_input)
            output = output.transpose(1,2)
            output = self.BN_layer(output)

            output = self.relu(output)
            output = torch.mean(output, dim=2)
            output = output + hs_ori
            
            g.ndata['hs'] = torch.squeeze(output)  

        if self.update_E:
            self.Upd_E(g)
        return g







class SGGN_layer_withoutE(nn.Module):

    def __init__(self, d_hidden, d_Conv, hp_Expand, max_degree, linear_agg, update_E):
        super().__init__()
        self.d_hidden = d_hidden
        self.Group_Gating = GatedCNN(
        d_model=self.d_hidden, # Model dimension d_model
        d_conv=d_Conv,    # Local convolution width
        expand=hp_Expand,    # Block expansion factor
        ).to("cuda")

        self.MP = Message_passing_layer_withoutE(max_degree, self.d_hidden)

        self.update_E = update_E
        if self.update_E:
            self.Upd_E = Updated_E_layer(d_hidden)
        
        self.BN_layer = nn.BatchNorm1d(self.d_hidden)

        self.relu = nn.ReLU6()

        self.linear_agg = linear_agg
        if linear_agg:
            self.linear_agg_layer = nn.Linear(max_degree+1, 1)


    def forward(self, g):
        
        hs_ori = g.ndata["hs"].float()
        hs = hs_ori.view(-1, 1, self.d_hidden)
        self.MP(g)     

        adj = g.adjacency_matrix()

        adj = torch.sparse_coo_tensor(
        adj.indices(),   
        adj.val,         
        size=adj.shape,  
        dtype=torch.float32).coalesce()
        n_nodes = adj.shape[0]
        eye_indices = torch.arange(n_nodes).unsqueeze(0).repeat(2,1).cuda()
        eye_values = torch.ones(n_nodes).cuda()
        new_indices = torch.cat([adj.indices(), eye_indices], dim=1)
        new_values = torch.cat([adj.values(), eye_values], dim = 0)
        adj = torch.sparse_coo_tensor(new_indices, new_values, (n_nodes, n_nodes)).coalesce().cuda()

        # print(adj_sparse)
        h = g.ndata["hs_aggregate"].float()

        h_input = torch.spmm(adj, h.view(n_nodes, -1))
        h_input = torch.cat((hs, h_input.view(n_nodes, -1, self.d_hidden)), dim = 1)


        if self.linear_agg:

            output = self.Group_Gating(h_input)
            output = output.transpose(1,2)
            output = self.BN_layer(output)

            output = self.relu(output)
            output = self.linear_agg_layer(output)
            output = output.transpose(1,2)
            output = output + hs
            
            g.ndata['hs'] = torch.squeeze(output)  


        else:

            output = self.Group_Gating(h_input)
            output = output.transpose(1,2)
            output = self.BN_layer(output)

            output = self.relu(output)
            output = torch.mean(output, dim=2)
            output = output + hs_ori
            
            g.ndata['hs'] = torch.squeeze(output)  

        if self.update_E:
            self.Upd_E(g)
        return g





class SGGN_layer_withoutC(nn.Module):

    def __init__(self, d_hidden, d_Conv, hp_Expand, max_degree, linear_agg, update_E):
        super().__init__()
        self.d_hidden = d_hidden
        self.Group_Gating = GatedCNN_withoutC(
        d_model=self.d_hidden, # Model dimension d_model
        d_conv=d_Conv,    # Local convolution width
        expand=hp_Expand,    # Block expansion factor
        ).to("cuda")

        self.MP = Message_passing_layer(max_degree, self.d_hidden)

        self.update_E = update_E
        if self.update_E:
            self.Upd_E = Updated_E_layer(d_hidden)
        
        self.BN_layer = nn.BatchNorm1d(self.d_hidden)

        self.relu = nn.ReLU6()

        self.linear_agg = linear_agg
        if linear_agg:
            self.linear_agg_layer = nn.Linear(max_degree+1, 1)


    def forward(self, g):
        
        hs_ori = g.ndata["hs"].float()
        hs = hs_ori.view(-1, 1, self.d_hidden)
        self.MP(g)     

        adj = g.adjacency_matrix()

        adj = torch.sparse_coo_tensor(
        adj.indices(),   
        adj.val,         
        size=adj.shape,  
        dtype=torch.float32).coalesce()
        n_nodes = adj.shape[0]
        eye_indices = torch.arange(n_nodes).unsqueeze(0).repeat(2,1).cuda()
        eye_values = torch.ones(n_nodes).cuda()
        new_indices = torch.cat([adj.indices(), eye_indices], dim=1)
        new_values = torch.cat([adj.values(), eye_values], dim = 0)
        adj = torch.sparse_coo_tensor(new_indices, new_values, (n_nodes, n_nodes)).coalesce().cuda()

        # print(adj_sparse)
        h = g.ndata["hs_aggregate"].float()

        h_input = torch.spmm(adj, h.view(n_nodes, -1))
        h_input = torch.cat((hs, h_input.view(n_nodes, -1, self.d_hidden)), dim = 1)


        if self.linear_agg:

            output = self.Group_Gating(h_input)
            output = output.transpose(1,2)
            output = self.BN_layer(output)

            output = self.relu(output)
            output = self.linear_agg_layer(output)
            output = output.transpose(1,2)
            output = output + hs
            
            g.ndata['hs'] = torch.squeeze(output)  


        else:

            output = self.Group_Gating(h_input)
            output = output.transpose(1,2)
            output = self.BN_layer(output)

            output = self.relu(output)
            output = torch.mean(output, dim=2)
            output = output + hs_ori
            
            g.ndata['hs'] = torch.squeeze(output)  

        if self.update_E:
            self.Upd_E(g)
        return g