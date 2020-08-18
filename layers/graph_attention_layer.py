import math

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
References: https://github.com/Diego999/pyGAT
"""

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, device):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        
        # Weight
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)).to(device)
        self.weight2 = nn.Parameter(torch.zeros(size=(2*out_features, 1))).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv, stdv)
                        
    def forward(self, x, adj):
        batch_size = x.size()[0]
        node_count = x.size()[1]
        
        x = x.reshape(batch_size * node_count, x.size()[2])
        x = torch.mm(x, self.weight)
        x = x.reshape(batch_size, node_count, self.weight.size()[-1])
        
        # Attention score
        attention_input = torch.cat([x.repeat(1, 1, node_count).view(batch_size, node_count * node_count, -1), x.repeat(1, node_count, 1)], dim=2).view(batch_size, node_count, -1, 2 * self.out_features)
        e = F.relu(torch.matmul(attention_input, self.weight2).squeeze(3))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        x = torch.bmm(attention, x)
        
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'