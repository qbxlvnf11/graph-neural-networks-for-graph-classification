import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.graph_convolution_layer import GraphConvolutionLayer

class GraphResNetLayer(nn.Module):
    def __init__(self, in_features, out_features, device):
        super(GraphResNetLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Graph convolution layer
        self.graph_convolution_layer = GraphConvolutionLayer(in_features, out_features, device)
        self.graph_convolution_layer2 = GraphConvolutionLayer(out_features, out_features, device)

    def forward(self, x, adj):
        x1 = F.relu(self.graph_convolution_layer(x, adj))
        x2 = self.graph_convolution_layer2(x, adj)
        
        # Add
        output = x + x2
        
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'