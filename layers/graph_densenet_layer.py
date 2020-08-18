import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.graph_convolution_layer import GraphConvolutionLayer

class GraphDenseNetLayer(nn.Module):
    def __init__(self, in_features, out_features, device, bias=True):
        super(GraphDenseNetLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Graph convolution layer
        self.graph_convolution_layer = GraphConvolutionLayer(in_features, out_features, device)
        self.graph_convolution_layer2 = GraphConvolutionLayer(in_features + out_features, out_features, device)

        # Pooling
        self.pooling = nn.Parameter(torch.FloatTensor(out_features * 4, out_features)).to(device)
        self.pooling_ration = 0.8
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.pooling.size(1))
        self.pooling.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, pooling):
        
        # Pooling
        if pooling:
            B, N, C = x.size()
            x = x.reshape(B * N, C)
            x = torch.mm(x, self.pooling).reshape(B, N, self.out_features)
            
        # Graph convolution layer
        x1 = self.graph_convolution_layer(x, adj)
        
        # Concat
        concat1 = torch.cat((x, x1), 2)
        
        # Graph convolution layer        
        x2 = self.graph_convolution_layer2(concat1, adj)
        
        # Concat
        concat2 = torch.cat((x, concat1, x2), 2)
        
        return concat2

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'