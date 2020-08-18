import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch.nn as nn
import torch.nn.functional as F

#from layers.graph_convolution_layer import GraphConvolutionLayer
from layers.graph_unet_layer import GraphUNetLayer
from readouts.basic_readout import readout_function

"""
Base paper: https://arxiv.org/pdf/1905.05178.pdf
"""

class GraphUNet(nn.Module):
    def __init__(self, n_feat, n_class, n_layer, agg_hidden, fc_hidden, dropout, readout, device):
        super(GraphUNet, self).__init__()
        
        self.n_layer = n_layer
        self.readout = readout
        
        # Pooling_rate
        pooling_rations = [0.8 - (i * 0.1) if i < 3 else 0.5 for i in range(n_layer)]
           
        # Graph unet layer
        self.graph_unet_layers = []
        for i in range(n_layer):
           if i == 0:
             self.graph_unet_layers.append(GraphUNetLayer(n_feat, agg_hidden, pooling_rations[i], device))
           else:
             self.graph_unet_layers.append(GraphUNetLayer(agg_hidden, agg_hidden, pooling_rations[i], device))
        
        # Fully-connected layer
        self.fc1 = nn.Linear(agg_hidden, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, n_class)

    def forward(self, data):
        
        for i in range(self.n_layer):
           # Graph unet layer
           data = self.graph_unet_layers[i](data)
        
        x = data[0]
                      
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
            layers += str(self.graph_unet_layers[i]) + '\n'
        layers += str(self.fc1) + '\n'
        layers += str(self.fc2) + '\n'
        return layers