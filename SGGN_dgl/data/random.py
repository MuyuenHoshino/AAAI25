import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np

import csv

import dgl

from scipy import sparse as sp
import numpy as np
import networkx as nx
import hashlib

import random
import torch.nn.functional as F
def seed_everything(seed):
    random.seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)


def laplace_decomp(g, max_freqs):


    # Laplacian
    n = g.number_of_nodes()
    # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    A = g.adj_external(scipy_fmt='csr')
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVals, EigVecs = np.linalg.eigh(L.toarray())
    EigVals, EigVecs = EigVals[: max_freqs], EigVecs[:, :max_freqs]  # Keep up to the maximum desired number of frequencies

    # Normalize and pad EigenVectors
    EigVecs = torch.from_numpy(EigVecs).float()
    EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)
    
    if n<max_freqs:
        g.ndata['EigVecs'] = F.pad(EigVecs, (0, max_freqs-n), value=float('nan'))
    else:
        g.ndata['EigVecs']= EigVecs
        
    
    #Save eigenvales and pad
    EigVals = torch.from_numpy(np.sort(np.abs(np.real(EigVals)))) #Abs value is taken because numpy sometimes computes the first eigenvalue approaching 0 from the negative
    
    if n<max_freqs:
        EigVals = F.pad(EigVals, (0, max_freqs-n), value=float('nan')).unsqueeze(0)
    else:
        EigVals=EigVals.unsqueeze(0)
        
    
    #Save EigVals node features
    g.ndata['EigVals'] = EigVals.repeat(g.number_of_nodes(),1).unsqueeze(2)
    
    return g



def make_full_graph(g):

    
    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))

    #Here we copy over the node feature data and laplace encodings
    full_g.ndata['feat'] = g.ndata['feat']

    try:
        full_g.ndata['EigVecs'] = g.ndata['EigVecs']
        full_g.ndata['EigVals'] = g.ndata['EigVals']
    except:
        pass
    
    #Populate edge features w/ 0s
    full_g.edata['feat']=torch.zeros(full_g.number_of_edges(), dtype=torch.long)
    full_g.edata['real']=torch.zeros(full_g.number_of_edges(), dtype=torch.long)
    
    #Copy real edge data over
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['feat'] = torch.ones(g.edata['feat'].shape[0], dtype=torch.long) 
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['real'] = torch.ones(g.edata['feat'].shape[0], dtype=torch.long) 
    
    return full_g


def init_positional_encoding(g, pos_enc_dim, type_init):
    """
        Initializing positional encoding with RWPE
    """
    
    n = g.number_of_nodes()

    if type_init == 'rand_walk':
        # Geometric diffusion features with Random Walk
        A = g.adj_external(scipy_fmt="csr")
        Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float) # D^-1
        RW = A * Dinv  
        M = RW
        
        # Iterate
        nb_pos_enc = pos_enc_dim
        PE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(nb_pos_enc-1):
            M_power = M_power * M
            PE.append(torch.from_numpy(M_power.diagonal()).float())
        PE = torch.stack(PE,dim=-1)
        g.ndata['pos_enc'] = PE  
    
    return g


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in random_Dataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']
    
    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)
    
    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g


# def make_full_graph(g):
#     """
#         Converting the given graph to fully connected
#         This function just makes full connections
#         removes available edge features 
#     """

#     full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))
#     full_g.ndata['feat'] = g.ndata['feat']
#     full_g.edata['feat'] = torch.zeros(full_g.number_of_edges()).long()
    
#     try:
#         full_g.ndata['lap_pos_enc'] = g.ndata['lap_pos_enc']
#     except:
#         pass

#     try:
#         full_g.ndata['wl_pos_enc'] = g.ndata['wl_pos_enc']
#     except:
#         pass    
    
#     return full_g



def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    A = g.adj_external(scipy_fmt='csr')
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    
    return g

def wl_positional_encoding(g):
    """
        WL-based absolute positional embedding 
        adapted from 
        
        "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
        Zhang, Jiawei and Zhang, Haopeng and Xia, Congying and Sun, Li, 2020
        https://github.com/jwzhanggy/Graph-Bert
    """
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
    node_list = g.nodes().numpy()

    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1


    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1
        
    g.ndata['wl_pos_enc'] = torch.LongTensor(list(node_color_dict.values()))
    return g









def generate_random_graph_with_features(num_graphs, avg_num_nodes, max_degree, num_features):
    graphs = []
    for _ in range(num_graphs):
        num_nodes = np.random.randint(avg_num_nodes // 1.5, avg_num_nodes * 1.5)  # Random number of nodes around the average
        degrees = np.random.randint(1, max_degree + 1, size=num_nodes)  # Random degrees for each node
        degrees = np.minimum(degrees, num_nodes - 1)  # Ensure degrees are not greater than number of nodes - 1
        # print("################################")
        # print(degrees)
        # print(type(degrees))
        edges = []
        for i, degree in enumerate(degrees):
            neighbors = np.random.choice(np.delete(np.arange(num_nodes), i), size=degree, replace=False)
            print("#####################")
            print(neighbors)
            edges.extend([(i, neighbor) for neighbor in neighbors])
        src, dst = zip(*edges)
        g = dgl.graph((src, dst), num_nodes=num_nodes)
        
        # Generate random node features
        features = torch.rand(num_nodes, num_features)
        g.ndata['feat'] = features
        g.edata['feat'] = torch.ones(g.number_of_edges(), 4)
        graphs.append(g)
    return graphs


def generate_random_dgl_graph_from_nx(num_graphs, avg_num_nodes, max_degree, num_features):
    graphs = []
    for _ in range(num_graphs):
        num_nodes = np.random.randint(avg_num_nodes -5, avg_num_nodes +5)

        # nx要求n * d必须是偶数，我们选择把节点个数固定为偶数
        if num_nodes % 2 != 0:  # 检查是否为奇数
            # 随机选择 +1 或 -1
            rand_choice = random.choice([-1, 1])
            num_nodes = num_nodes + rand_choice

        degree = np.random.randint( 1, max_degree+1)

        G = nx.random_regular_graph(degree, num_nodes)
        # 将NetworkX图转换为DGL图
        dgl_G = dgl.from_networkx(G)
        features = torch.rand(num_nodes, num_features)
        dgl_G.ndata['feat'] = features
        dgl_G.edata['feat'] = torch.ones(dgl_G.number_of_edges())
        graphs.append(dgl_G)
        
    return graphs






class random_Dataset_dgl(torch.utils.data.Dataset):
    def __init__(self, num_graphs = 16, avg_num_nodes = 1000, max_degree = 5, num_features = 32):

        self.num_graphs = num_graphs

        self.graph_lists = generate_random_dgl_graph_from_nx(num_graphs, avg_num_nodes, max_degree, num_features)
        self.labels = torch.rand(num_graphs, 1)

    def __getitem__(self, idx):
        return self.graph_lists[idx], self.labels[idx]
    def __len__(self):
        return self.num_graphs


class randomDataset(torch.utils.data.Dataset):

    def __init__(self, num_nodes):
        """
            Loading ZINC dataset
        """
        
        seed_everything(12345)
        
        start = time.time()
        # print(num_nodes)
        self.train = random_Dataset_dgl(num_graphs=16, avg_num_nodes=num_nodes)
        self.val = random_Dataset_dgl(num_graphs=0, avg_num_nodes=num_nodes)
        self.test = random_Dataset_dgl(num_graphs=0, avg_num_nodes=num_nodes)

        max_degree = 0
        mean_degrees = []
        for tu in (self.train + self.val + self.test):
            g, l = tu
            in_degrees = g.in_degrees()
            # out_degrees = g.out_degrees()
            #print(in_degrees)
            temp,_ = in_degrees.max(dim = 0)
            
            # print(in_degrees)
            # print(type(in_degrees))
            mean_degree = torch.mean(in_degrees.float())
            mean_degrees.append(mean_degree.float())

            if temp > max_degree:
                max_degree = temp


        print(int(max_degree))   
        # print(np.mean(mean_degrees))


        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        # print(labels[0].shape)
        labels = torch.cat(labels).unsqueeze(1)
        
        '''
        different numpy version
        '''


        batched_graph = dgl.batch(graphs)       
        
        return batched_graph, labels
    
    
    def _add_self_loops(self):
        
        # function for adding self loops
        # this function will be called only if self_loop flag is True
            
        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]

    def _make_full_graph(self):
        
        # function for converting graphs to full graphs
        # this function will be called only if full_graph flag is True
        self.train.graph_lists = [make_full_graph(g) for g in self.train.graph_lists]
        self.val.graph_lists = [make_full_graph(g) for g in self.val.graph_lists]
        self.test.graph_lists = [make_full_graph(g) for g in self.test.graph_lists]
    
    
    def _add_laplacian_positional_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
        self.test.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]

    def _add_wl_positional_encodings(self):
        
        # WL positional encoding from Graph-Bert, Zhang et al 2020.
        self.train.graph_lists = [wl_positional_encoding(g) for g in self.train.graph_lists]
        self.val.graph_lists = [wl_positional_encoding(g) for g in self.val.graph_lists]
        self.test.graph_lists = [wl_positional_encoding(g) for g in self.test.graph_lists]

    def _init_positional_encodings(self, pos_enc_dim):
        type_init = "rand_walk"
        # Initializing positional encoding randomly with l2-norm 1
        self.train.graph_lists = [init_positional_encoding(g, pos_enc_dim, type_init) for g in self.train.graph_lists]
        self.val.graph_lists = [init_positional_encoding(g, pos_enc_dim, type_init) for g in self.val.graph_lists]
        self.test.graph_lists = [init_positional_encoding(g, pos_enc_dim, type_init) for g in self.test.graph_lists]

    def _laplace_decomp(self, max_freqs):
        self.train.graph_lists = [laplace_decomp(g, max_freqs) for g in self.train.graph_lists]
        self.val.graph_lists = [laplace_decomp(g, max_freqs) for g in self.val.graph_lists]
        self.test.graph_lists = [laplace_decomp(g, max_freqs) for g in self.test.graph_lists]