import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch.nn as nn
import torch.nn.functional as F

from layers.graph_convolution_layer import GraphConvolutionLayer
from readouts.basic_readout import readout_function

"""
Base paper: https://arxiv.org/abs/1609.02907
"""

class GCN(nn.Module):
    def __init__(self, n_feat, n_class, n_layer, agg_hidden, fc_hidden, dropout, readout, device):
        super(GCN, self).__init__()
        
        self.n_layer = n_layer
        self.dropout = dropout
        self.readout = readout
        
        # Graph convolution layer
        self.graph_convolution_layers = []
        for i in range(n_layer):
           if i == 0:
             self.graph_convolution_layers.append(GraphConvolutionLayer(n_feat, agg_hidden, device))
           else:
             self.graph_convolution_layers.append(GraphConvolutionLayer(agg_hidden, agg_hidden, device))
        
        # Fully-connected layer
        self.fc1 = nn.Linear(agg_hidden, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, n_class)
    
    def forward(self, data):
        x, adj = data[:2]

        for i in range(self.n_layer):
           # Graph convolution layer
           x = F.relu(self.graph_convolution_layers[i](x, adj))
                      
           # Dropout
           if i != self.n_layer - 1:
             x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Readout
        x = readout_function(x, self.readout)
        
        # Fully-connected layer
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))

        return x
        
    def __repr__(self):
        layers = ''
        
        for i in range(self.n_layer):
            layers += str(self.graph_convolution_layers[i]) + '\n'
        layers += str(self.fc1) + '\n'
        layers += str(self.fc2) + '\n'
        return layers
            
            