import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_scatter import scatter, scatter_add

from collections import Counter
from tqdm import tqdm
from torch_geometric.nn import knn

from torch_geometric.utils import degree

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output




class GatedCNN(nn.Module):

    def __init__(self, d_model, d_conv, expand, layer_idx=0, **kwargs):
        super().__init__()

        hidden = d_model
        d_inner = hidden * expand

        self.fc1 = nn.Linear(hidden, hidden * expand * 2)
        # self.act = nn.GELU()
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
        x = self.conv1d(x)[:, :, :L] # depthwise convolution over time, with a short filter
        x = x.transpose(1, 2) # (B, L, ED)
        
        x = self.act(x)
        z = self.act(z)
        
        output = x * z
        output = self.fc2(output) # (B, L, D)

        output = self.norm(output) + x_ori

        return output



class GatedCNN_withoutC(nn.Module):

    def __init__(self, d_model, d_conv, expand, layer_idx=0, **kwargs):
        super().__init__()

        hidden = d_model
        d_inner = hidden * expand

        self.fc1 = nn.Linear(hidden, hidden * expand * 2)
        # self.act = nn.GELU()
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




class SGGN(pyg_nn.conv.MessagePassing):
    def __init__(self, d_NodeAttr, max_degree = 1, dropout=0):
        super(SGGN, self).__init__(aggr='mean')
        '''
        max_degree: 
            pep: 5
            pcqm: 6
        '''
        self.max_degree = max_degree
        self.d_NodeAttr = d_NodeAttr


        self.act = nn.Sigmoid()


        self.dropout = dropout

        self.update_e = nn.Linear(d_NodeAttr, d_NodeAttr)


        self.Group_Gated = GatedCNN(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=self.d_NodeAttr, # Model dimension d_model
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
        # layer_idx = layer_idx
        ).to("cuda")

        self.agg_linear = nn.Linear(self.max_degree + 1, 1)

        self.BN_layer = nn.BatchNorm1d(self.d_NodeAttr)

    def forward(self, batch):

        x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index

        self.n_nodes = x.shape[0] 


        x_out = self.propagate(edge_index = edge_index, x=x, edge_attr = e)
        batch.x = x_out

        return batch


    

    def aggregate(self, edge_attr, edge_index, x):


        src_index = edge_index[0]
        dst_index = edge_index[1]


        # Determine unique labels and the max count for padding
        unique_labels, inverse_indices = torch.unique(src_index, return_inverse=True)
 
        max_count = torch.bincount(inverse_indices).max().item()

        output_tensor = torch.full((self.n_nodes, self.max_degree, self.d_NodeAttr), 0).float().cuda()

        tobe_degree_tensor = torch.full((self.n_nodes, self.max_degree), self.max_degree * 2).long().cuda()
        all_degree = degree(dst_index,
                   num_nodes=self.n_nodes, dtype=torch.long)

        indices = torch.argsort(inverse_indices)
        sorted_inverse_indices = inverse_indices[indices]
        sorted_values = dst_index[indices]

        msg = x[dst_index][indices]
        # print(msg.shape)
        position = torch.cumsum(torch.nn.functional.one_hot(sorted_inverse_indices), dim=0).cuda() - 1 

        output_tensor[unique_labels[sorted_inverse_indices], position[torch.arange(len(position)), sorted_inverse_indices]] = msg


        tobe_degree_tensor[unique_labels[sorted_inverse_indices], position[torch.arange(len(position)), sorted_inverse_indices]] = all_degree[sorted_values]

        tobe_degree_tensor = tobe_degree_tensor.unsqueeze(-1)

        updated_x = output_tensor

        B, n_neibor, D = updated_x.shape
        combined_tensor = torch.cat((updated_x, tobe_degree_tensor), dim=2)
        sorted_tensor, sorted_indices = torch.sort(combined_tensor[:, :, -1], dim=1)
        updated_x = torch.gather(updated_x, 1, sorted_indices.unsqueeze(-1).expand(-1, n_neibor, D))


        updated_x = scatter(updated_x[src_index], dst_index, 0, None, self.n_nodes, reduce='sum')
        updated_x = torch.cat((updated_x, x.view(self.n_nodes, 1, self.d_NodeAttr)), dim = 1 )


        updated_x = self.Group_Gated(updated_x)

        updated_x = updated_x.transpose(1,2)

        updated_x = self.BN_layer(updated_x)
        updated_x = self.act(updated_x)
        updated_x = self.agg_linear(updated_x)

        updated_x = updated_x.transpose(1,2)


        return torch.squeeze(updated_x)

    def update(self, aggr_out, x):

        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """

        x = x + aggr_out

        return x





class SGGN_without_sort(pyg_nn.conv.MessagePassing):
    def __init__(self, d_NodeAttr, max_degree = 1, dropout=0):
        super(SGGN_without_sort, self).__init__(aggr='mean')
        '''
        max_degree: 
            pep: 5
            voc: 32
            pcqm: 6
        '''
        self.max_degree = max_degree
        self.d_NodeAttr = d_NodeAttr

        self.act = nn.Sigmoid()

        self.dropout = dropout

        self.Group_Gated = GatedCNN(
        d_model=self.d_NodeAttr, # Model dimension d_model
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
        # layer_idx = layer_idx
        ).to("cuda")

        self.agg_linear = nn.Linear(self.max_degree + 1, 1)
        self.BN_layer = nn.BatchNorm1d(self.d_NodeAttr)

    def forward(self, batch):

        x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index

        self.n_nodes = x.shape[0] 

        x_out = self.propagate(edge_index = edge_index, x=x, edge_attr = e)
        batch.x = x_out

        return batch


    

    def aggregate(self, edge_attr, edge_index, x):

        src_index = edge_index[0]
        dst_index = edge_index[1]

        unique_labels, inverse_indices = torch.unique(src_index, return_inverse=True)

 
        max_count = torch.bincount(inverse_indices).max().item()

        output_tensor = torch.full((self.n_nodes, self.max_degree, self.d_NodeAttr), 0).float().cuda()

        tobe_degree_tensor = torch.full((self.n_nodes, self.max_degree), self.max_degree * 2).long().cuda()
        all_degree = degree(dst_index,
                   num_nodes=self.n_nodes, dtype=torch.long)

        indices = torch.argsort(inverse_indices)
        sorted_inverse_indices = inverse_indices[indices]
        sorted_values = dst_index[indices]

        msg = x[dst_index][indices]
        # print(msg.shape)
        position = torch.cumsum(torch.nn.functional.one_hot(sorted_inverse_indices), dim=0).cuda() - 1 

        output_tensor[unique_labels[sorted_inverse_indices], position[torch.arange(len(position)), sorted_inverse_indices]] = msg


        updated_x = output_tensor

        updated_x = scatter(updated_x[src_index], dst_index, 0, None, self.n_nodes, reduce='sum')
        updated_x = torch.cat((updated_x, x.view(self.n_nodes, 1, self.d_NodeAttr)), dim = 1 )


        updated_x = self.Group_Gated(updated_x)

        updated_x = updated_x.transpose(1,2)

        updated_x = self.BN_layer(updated_x)
        updated_x = self.act(updated_x)
        updated_x = self.agg_linear(updated_x)

        updated_x = updated_x.transpose(1,2)


        return torch.squeeze(updated_x)

    def update(self, aggr_out, x):

        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """

        x = x + aggr_out

        return x





class SGGN_without_sort_withoutC(pyg_nn.conv.MessagePassing):
    def __init__(self, d_NodeAttr, max_degree = 1, dropout=0):
        super(SGGN_without_sort_withoutC, self).__init__(aggr='mean')
        '''
        max_degree: 
            pep: 5
            voc: 32
            pcqm: 6
        '''
        self.max_degree = max_degree
        self.d_NodeAttr = d_NodeAttr

        self.act = nn.Sigmoid()

        self.dropout = dropout

        self.Group_Gated = GatedCNN_withoutC(
        d_model=self.d_NodeAttr, # Model dimension d_model
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
        # layer_idx = layer_idx
        ).to("cuda")

        self.agg_linear = nn.Linear(self.max_degree + 1, 1)
        self.BN_layer = nn.BatchNorm1d(self.d_NodeAttr)

    def forward(self, batch):

        x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index

        self.n_nodes = x.shape[0] 

        x_out = self.propagate(edge_index = edge_index, x=x, edge_attr = e)
        batch.x = x_out

        return batch


    

    def aggregate(self, edge_attr, edge_index, x):

        src_index = edge_index[0]
        dst_index = edge_index[1]

        unique_labels, inverse_indices = torch.unique(src_index, return_inverse=True)

 
        max_count = torch.bincount(inverse_indices).max().item()

        output_tensor = torch.full((self.n_nodes, self.max_degree, self.d_NodeAttr), 0).float().cuda()

        tobe_degree_tensor = torch.full((self.n_nodes, self.max_degree), self.max_degree * 2).long().cuda()
        all_degree = degree(dst_index,
                   num_nodes=self.n_nodes, dtype=torch.long)

        indices = torch.argsort(inverse_indices)
        sorted_inverse_indices = inverse_indices[indices]
        sorted_values = dst_index[indices]

        msg = x[dst_index][indices]
        # print(msg.shape)
        position = torch.cumsum(torch.nn.functional.one_hot(sorted_inverse_indices), dim=0).cuda() - 1 

        output_tensor[unique_labels[sorted_inverse_indices], position[torch.arange(len(position)), sorted_inverse_indices]] = msg


        updated_x = output_tensor

        updated_x = scatter(updated_x[src_index], dst_index, 0, None, self.n_nodes, reduce='sum')
        updated_x = torch.cat((updated_x, x.view(self.n_nodes, 1, self.d_NodeAttr)), dim = 1 )


        updated_x = self.Group_Gated(updated_x)

        updated_x = updated_x.transpose(1,2)

        updated_x = self.BN_layer(updated_x)
        updated_x = self.act(updated_x)
        updated_x = self.agg_linear(updated_x)

        updated_x = updated_x.transpose(1,2)


        return torch.squeeze(updated_x)

    def update(self, aggr_out, x):

        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """

        x = x + aggr_out

        return x





















@register_layer('sggnconv')
class SGGN_Conv(nn.Module):

    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()


        # self.model = Mamba_MP(max_degree = 3, 
        #                       d_NodeAttr = 64,
        #                       residual=True,**kwargs
        #                       )

    def forward(self, batch):
        return self.model(batch)
