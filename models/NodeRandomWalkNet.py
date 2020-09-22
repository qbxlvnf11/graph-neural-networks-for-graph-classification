import networkx as nx
from node2vec import Node2Vec
import numpy as np
import operator
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from readouts.basic_readout import readout_function
from datasets.graph_node_random_walk import get_node_random_walk

class NodeRandomWalkNet(nn.Module):
    def __init__(self, n_feat, n_class, n_layer, agg_hidden, fc_hidden, dropout, readout, device,
    walk_length, node_random_walk_model_name):
        super(NodeRandomWalkNet, self).__init__()
        
        self.n_layer = n_layer
        self.dropout = dropout
        self.readout = readout
        self.device = device
        self.node_random_walk_model_name = node_random_walk_model_name
        
        # LSTM
        if node_random_walk_model_name == 'LSTM':
            self.lstm_layers = nn.LSTM(input_size=walk_length, hidden_size=agg_hidden, num_layers=n_layer)
        # GRU
        elif node_random_walk_model_name == 'GRU':
            self.gru_layers = nn.GRU(input_size=walk_length, hidden_size=agg_hidden, num_layers=n_layer)        
                
        # Fully-connected layer
        self.fc1 = nn.Linear(agg_hidden, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, n_class)
    
    def forward(self, data):
        x, adj = data[:2]
        random_walks = data[6]
        random_walks = random_walks.reshape(random_walks.size()[0] * random_walks.size()[1], random_walks.size()[2], random_walks.size()[3])
        
        # LSTM
        if self.node_random_walk_model_name == 'LSTM':
            x = self.lstm_layers(random_walks)[0]
        # GRU
        elif self.node_random_walk_model_name == 'GRU':
            x = self.gru_layers(random_walks)[0]     
                    
        x = torch.mean(x, dim=1).squeeze()
        x = x.reshape(adj.size()[0], adj.size()[1], x.size()[-1])
        
        # Readout
        x = readout_function(x, self.readout)
        
        # Fully-connected layer
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))

        return x
        
    def __repr__(self):
        layers = ''
        if self.node_random_walk_model_name == 'LSTM':
            layers += str(self.lstm_layers) + '\n'
        elif self.node_random_walk_model_name == 'GRU':
            layers += str(self.gru_layers) + '\n'
        layers += str(self.fc1) + '\n'
        layers += str(self.fc2) + '\n'
        return layers
            
            