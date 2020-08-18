import math

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
References: https://github.com/williamleif/graphsage-simple
"""

class MLP(nn.Module):
    def __init__(self, n_layer, input_dim, hidden_dim, output_dim):

        super(MLP, self).__init__()

        self.linear_or_not = True # Default is linear model
        self.n_layer = n_layer

        if n_layer < 1:
            raise ValueError("number of layers should be positive!")
        elif n_layer == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(n_layer - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(n_layer - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.n_layer - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.n_layer - 1](h)

class GraphIsomorphismLayer(nn.Module):
    def __init__(self, layer_num, in_features, out_features, neighbor_pooling_type, learn_eps, eps, device, num_mlp_layers = 2):
        super(GraphIsomorphismLayer, self).__init__()
        self.layer_num = layer_num
        self.in_features = in_features
        self.out_features = out_features 
        
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = eps
        self.device = device

        self.mlp = MLP(num_mlp_layers, in_features, out_features, out_features).to(device)

        self.batch_norm = nn.BatchNorm1d(out_features).to(device)

    def maxpool(self, h, padded_neighbor_list):
        ###Element-wise minimum will never affect max-pooling

        dummy = torch.min(h, dim = 0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim = 1)[0]
        return pooled_re

    # Pooling neighboring nodes and center nodes separately by epsilon reweighting   
    def next_layer_eps(self, h, padded_neighbor_list = None, Adj_block = None):
        if self.neighbor_pooling_type == "max":
            # If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        # Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[self.layer_num]) * h
        pooled_rep = self.mlp(pooled)
        h = self.batch_norm(pooled_rep)

        # Non-linearity
        h = F.relu(h)
        return h

    # Pooling neighboring nodes and center nodes altogether  
    def next_layer(self, h, padded_neighbor_list = None, Adj_block = None):       
        if self.neighbor_pooling_type == "max":
            # If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                #If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        # Representation of neighboring and center nodes 
        pooled_rep = self.mlp(pooled)
        h = self.batch_norm(pooled_rep)

        # Non-linearity
        h = F.relu(h)
        return h
        
    def forward(self, h, Adj_block, padded_neighbor_list):
        if self.neighbor_pooling_type == "max" and self.learn_eps:
            h = self.next_layer_eps(h, padded_neighbor_list = padded_neighbor_list)
        elif not self.neighbor_pooling_type == "max" and self.learn_eps:
            h = self.next_layer_eps(h, Adj_block = Adj_block)
        elif self.neighbor_pooling_type == "max" and not self.learn_eps:
            h = self.next_layer(h, padded_neighbor_list = padded_neighbor_list)
        elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
            h = self.next_layer(h, Adj_block = Adj_block)
        
        return h

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'