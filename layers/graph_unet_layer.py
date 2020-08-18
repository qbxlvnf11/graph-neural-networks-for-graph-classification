import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.graph_convolution_layer import GraphConvolutionLayer

"""
References: https://github.com/bknyaz/graph_nn
"""

class GraphUNetLayer(nn.Module):
    def __init__(self, in_features, out_features, pooling_ration, device):
        super(GraphUNetLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
                
        # Graph convolution layer
        self.graph_convolution_layer = GraphConvolutionLayer(in_features, out_features, device)
        
        # Pooling
        self.pooling = nn.Parameter(torch.FloatTensor(out_features, 1)).to(device)
        self.pooling_ration = pooling_ration
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.pooling.size(1))
        self.pooling.data.uniform_(-stdv, stdv)
        
    def forward(self, data):
        x, adj = data[:2]
        n_nodes = data[3]
        mask = data[2].clone()
        
        # Graph convolution layer
        x =  self.graph_convolution_layer(x, adj)
        
        # Pooling
        B, N, C = x.shape
        y = torch.mm(x.reshape(B * N, C), self.pooling).reshape(B, N)
        
        # Node scores used for ranking below
        y = y / (torch.sum(self.pooling ** 2).reshape(1, 1) ** 0.5)
        idx = torch.sort(y, dim=1)[1]
        
        n_remove = (n_nodes.float() * (1 - self.pooling_ration)).long()
        
        n_nodes_prev = n_nodes
        n_nodes = n_nodes - n_remove
                    
        for b in range(B):
            idx_b = idx[b, mask[b, idx[b]] == 1]
            mask[b, idx_b[:n_remove[b]]] = 0
                
        for b in range(B):
            s = torch.sum(y[b] >= torch.min((y * mask.float())[b]))
        
        mask = mask.unsqueeze(2)
        x = x * torch.tanh(y).unsqueeze(2) * mask
        adj = mask * adj * mask.reshape(B, 1, N)
        mask = mask.squeeze()
        data = (x, adj, mask, n_nodes)
        
        return data
        
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'