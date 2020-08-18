import networkx as nx
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.graph_isomorphism_layer import GraphIsomorphismLayer
from readouts.basic_readout import readout_function

"""
Base paper: https://arxiv.org/pdf/1810.00826.pdf
"""

class GIN(nn.Module):
    def __init__(self, n_feat, n_class, n_layer, agg_hidden, fc_hidden, dropout, readout, device, neighbor_pooling_type = 'sum', graph_pooling_type = 'sum', learn_eps = True):
        super(GIN, self).__init__()
        
        self.n_layer = n_layer
        self.dropout = dropout
        self.readout = readout
        self.device = device
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        #self.graph_pooling_type = graph_pooling_type
        eps = nn.Parameter(torch.zeros(n_layer)).to(device)
        
        # Graph convolution layer
        self.graph_isomorphism_layers = []
        for i in range(n_layer):
            if i == 0:
                self.graph_isomorphism_layers.append(GraphIsomorphismLayer(i, n_feat, agg_hidden, neighbor_pooling_type, learn_eps, eps, device))
            else:
                self.graph_isomorphism_layers.append(GraphIsomorphismLayer(i, agg_hidden, agg_hidden, neighbor_pooling_type, learn_eps, eps, device))

        # List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()
        
        # Fully-connected layer
        self.fc1 = nn.Linear(agg_hidden, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, n_class)
        
        #Linear function that maps the hidden representation at dofferemt layers into a prediction score
        #self.linears_prediction = torch.nn.ModuleList()
        #for layer in range(n_layer + 1):
            #if layer == 0:
                #self.linears_prediction.append(nn.Linear(n_feat, n_class))
            #else:
                #self.linears_prediction.append(nn.Linear(agg_hidden, n_class))
                
    def __get_edge_adj_matrix(self, adj_list):
        edge_adj_list = []
        
        for adj in adj_list:
            G = nx.Graph(adj.to('cpu').numpy())
            edges = [list(pair) for pair in G.edges]
            edges.extend([[i, j] for j, i in edges])
            
            edge_adj_list.append(torch.LongTensor(edges).transpose(0,1))
            
        return edge_adj_list

    def __get_node_neighbors(self, adj_list):
        node_neighbors_list = []
        
        for adj in adj_list:
            G = nx.Graph(adj.to('cpu').numpy())
            neighbors = [[] for i in range(adj.size()[1])]
            for i, j in G.edges:
                neighbors[i].append(j)
                neighbors[j].append(i)
                
            node_neighbors_list.append(neighbors)
        
        return node_neighbors_list
        
    # Create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)
    def __preprocess_graphpool(self, x_list):
        
        start_idx = [0]

        #compute the padded neighbor list
        for i, x in enumerate(x_list):
            start_idx.append(start_idx[i] + x.size()[0])

        idx = []
        elem = []
        for i, x in enumerate(x_list):
            # Average pooling
            if self.graph_pooling_type == "average":
                elem.extend([1./x.size()[0]]*x.size()[0])
            
            # Sum pooling            
            else:
                elem.extend([1]*x.size()[0])

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0,1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(x_list), start_idx[-1]]))
        
        return graph_pool.to(self.device)
           
    # Create padded_neighbor_list in concatenated graph
    def __preprocess_neighbors_maxpool(self, x_list, neighbors_list, max_neighbor_list):
        # Compute the maximum number of neighbors within the graphs in the current minibatch
        max_deg = max([max_neighbor for max_neighbor in max_neighbor_list])

        padded_neighbor_list = []
        start_idx = [0]

        for i , x in enumerate(x_list):
            start_idx.append(start_idx[i] + x.size()[0])
            padded_neighbors = []
            for j in range(len(neighbors_list[i])):
                # Add off-set values to the neighbor indices
                pad = [n + start_idx[i] for n in neighbors_list[i][j]]
                # Padding, dummy data is assumed to be stored in -1
                pad.extend([-1]*(max_deg - len(pad)))

                # Add center nodes in the maxpooling if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
                if not self.learn_eps:
                    pad.append(j + start_idx[i])

                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.FloatTensor(padded_neighbor_list)

    # Create block diagonal sparse matrix
    def __preprocess_neighbors_sumavepool(self, x_list, edge_adj_list):
        edge_mat_list = []
        start_idx = [0]
        
        for i, x in enumerate(x_list):
            start_idx.append(start_idx[i] + x.size()[0])
            edge_mat_list.append(edge_adj_list[i] + start_idx[i])
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        # Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
        if not self.learn_eps:
            num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
            elem = torch.ones(num_node)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1],start_idx[-1]]))

        return Adj_block.to(self.device)
    
    def forward(self, data):
        x_list = data[0]
        adj_list = data[1]
        edge_adj_list = self.__get_edge_adj_matrix(adj_list)
        neighbors_list = self.__get_node_neighbors(adj_list)
        max_neighbor_list = data[5]
        
        x_concat = torch.cat([x for x in x_list], 0).to(self.device)
        #graph_pool = self.__preprocess_graphpool(x_list)
        
        padded_neighbor_list = None
        Adj_block = None
        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(x_list, neighbors_list, max_neighbor_list)
        else:
            Adj_block = self.__preprocess_neighbors_sumavepool(x_list, edge_adj_list)
            
        # List of hidden representation at each layer (including input)
        hidden_rep = [x_concat]
        h = x_concat
        
        for i in range(self.n_layer):
           # Graph isomorphism layer
           h = self.graph_isomorphism_layers[i](h, Adj_block, padded_neighbor_list)
           
           # Dropout           
           h = F.dropout(h, p=self.dropout, training=self.training)
           
           #hidden_rep.append(h)
        
        # Perform pooling over all nodes in each graph in every layer
        #score_over_layer = 0
        #for layer, h in enumerate(hidden_rep):
            #pooled_h = torch.spmm(graph_pool, h)
            #score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.dropout, training = self.training)

        #return score_over_layer          
        
        h = h.reshape(x_list.size()[0], x_list.size()[1], h.size()[-1])
                
        # Readout
        h = readout_function(h, self.readout)
        
        # Fully-connected layer
        h = F.relu(self.fc1(h))
        h = F.softmax(self.fc2(h))

        return h
        
    def __repr__(self):
        layers = ''
        
        for i in range(self.n_layer):
            layers += str(self.graph_isomorphism_layers[i]) + '\n'
        layers += str(self.fc1) + '\n'
        layers += str(self.fc2) + '\n'
        return layers