import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch.nn as nn
import torch.nn.functional as F

from layers.graph_densenet_layer import GraphDenseNetLayer
from layers.graph_convolution_layer import GraphConvolutionLayer
from readouts.basic_readout import readout_function

class GraphDenseNet(nn.Module):
    def __init__(self, n_feat, n_class, n_layer, agg_hidden, fc_hidden, dropout, readout, device):
        super(GraphDenseNet, self).__init__()
        
        self.n_layer = n_layer
        self.dropout = dropout
        self.readout = readout
        
        # Graph convolution layer
        self.graph_convolution_layer = GraphConvolutionLayer(n_feat, agg_hidden, device)
        
        # Graph densenet layer
        self.graph_densenet_layers = []
        for i in range(self.n_layer):
            self.graph_densenet_layers.append(GraphDenseNetLayer(agg_hidden, agg_hidden, device))
        
        # Fully-connected layer
        self.fc1 = nn.Linear((agg_hidden * 4), fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, n_class)
    
    def forward(self, data):
        x, adj = data[:2]
        
        # Graph convolution layer
        x = F.relu(self.graph_convolution_layer(x, adj))

        # Dropout
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i in range(self.n_layer):
           pooling = False
           if i != 0: pooling = True
           
           # Graph densenet layer
           x = F.relu(self.graph_densenet_layers[i](x, adj, pooling))
                      
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
        
        layers += str(self.graph_convolution_layer) + '\n'
        for i in range(self.n_layer):
            layers += str(self.graph_densenet_layers[i]) + '\n'
        layers += str(self.fc1) + '\n'
        layers += str(self.fc2) + '\n'
        return layers
            
            