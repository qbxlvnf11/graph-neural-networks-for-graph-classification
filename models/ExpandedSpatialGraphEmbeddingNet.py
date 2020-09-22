import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from readouts.basic_readout import readout_function
from layers.graph_convolution_layer import GraphConvolutionLayer

class ExpandedSpatialGraphEmbeddingNet(nn.Module):
    def __init__(self, n_class, agg_hidden, fc_hidden, dropout, device, readout,  
        walk_length, n_node_random_walk_model_layer, n_spatial_graph_embedding_model_layer,
        node_random_walk_model_name, spatial_graph_embedding_model_name = 'GCN'):
        super(ExpandedSpatialGraphEmbeddingNet, self).__init__()
        
        self.dropout = dropout
        self.readout = readout
        self.device = device
        self.n_spatial_graph_embedding_model_layer = n_spatial_graph_embedding_model_layer
        self.node_random_walk_model_name = node_random_walk_model_name
        
        # Node random walk model
        if node_random_walk_model_name == 'LSTM':
            self.lstm_layers = nn.LSTM(input_size=walk_length, hidden_size=agg_hidden, num_layers=n_node_random_walk_model_layer)
        elif node_random_walk_model_name == 'GRU':
            self.gru_layers = nn.GRU(input_size=walk_length, hidden_size=agg_hidden, num_layers=n_node_random_walk_model_layer)

        # Spatial graph embedding model
        if spatial_graph_embedding_model_name == 'GCN':
            self.graph_convolution_layers = []
            for i in range(n_spatial_graph_embedding_model_layer):
                self.graph_convolution_layers.append(GraphConvolutionLayer(agg_hidden, agg_hidden, device))
                
        # Fully-connected layer
        self.fc1 = nn.Linear(agg_hidden, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, n_class)
           
    def forward(self, data, fold_id, mode):
        x, adj = data[:2]
        random_walks = data[6]
        random_walks = random_walks.reshape(random_walks.size()[0] * random_walks.size()[1], random_walks.size()[2], random_walks.size()[3])
        
        # Node random walk model
        if self.node_random_walk_model_name == 'LSTM':
            x = self.lstm_layers(random_walks)[0]
        elif self.node_random_walk_model_name == 'GRU':
            x = self.gru_layers(random_walks)[0]  
        
        x = torch.mean(x, dim=1).squeeze()
        x = x.reshape(adj.size()[0], adj.size()[1], x.size()[-1])
        
        # Spatial graph embedding model
        for i in range(self.n_spatial_graph_embedding_model_layer):
           x = F.relu(self.graph_convolution_layers[i](x, adj))
                      
           # Dropout
           if i != self.n_spatial_graph_embedding_model_layer - 1:
             x = F.dropout(x, p=self.dropout, training=self.training)        
        
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
        for i in range(self.n_spatial_graph_embedding_model_layer):
            layers += str(self.graph_convolution_layers[i]) + '\n'
        layers += str(self.fc1) + '\n'
        layers += str(self.fc2) + '\n'
        return layers
###

# class ExpandedSpatialGraphEmbeddingNet(nn.Module):
#     def __init__(self, n_class, fc_hidden, dropout, device,
#         n_folds, dataset_name, readout,  
#         n_node_random_walk_model_layer = 1, hidden_node_random_walk_model = 64, n_spatial_graph_embedding_model_layer = 1, hidden_spatial_graph_embedding_model = 64,
#         spatial_graph_embedding_model_name = 'GCN', freeze_layer = True, fc_layer_type = 'A', fc_dropout = 0.2):
#         super(ExpandedSpatialGraphEmbeddingNet, self).__init__()
        
#         self.dropout = dropout
#         self.fc_dropout = fc_dropout
#         self.readout = readout
#         self.spatial_graph_embedding_model_name = spatial_graph_embedding_model_name
#         self.n_spatial_graph_embedding_model_layer = n_spatial_graph_embedding_model_layer
#         self.fc_layer_type = fc_layer_type
        
#         basic_path = './save_model/'
        
#         # Load node random walk model each fold
#         print('Load node random walk model ...')
        
#         self.node_random_walk_models = []
#         for i in range(n_folds):
#             node_random_walk_model_name = 'NodeRandomWalkNet_' + dataset_name + '_' + readout + '_' + str(i) + '_' + str(n_node_random_walk_model_layer) + '_h' + str(hidden_node_random_walk_model) + '.pt'
#             node_random_walk_model = torch.load(basic_path + 'NodeRandomWalkNet/' + node_random_walk_model_name).to(device)
            
#             if freeze_layer:
#                 for child in node_random_walk_model.children():
#                     for param in child.parameters():
#                         param.requires_grad = False
                
#             self.node_random_walk_models.append(node_random_walk_model)

#         print('node_random_walk_model')
#         print(self.node_random_walk_models[0])

#         # Load spatial graph embedding model each fold
#         print('Load spatial graph embedding model ...')
        
#         self.spatial_graph_embedding_models = []
#         for i in range(n_folds):
#             spatial_graph_embedding_model_name = self.spatial_graph_embedding_model_name + '_' + dataset_name + '_' + readout + '_' + str(i) + '_' + str(n_spatial_graph_embedding_model_layer + 1) + '_h' + str(hidden_spatial_graph_embedding_model) + '.pt'
#             spatial_graph_embedding_model = torch.load(basic_path + self.spatial_graph_embedding_model_name + '/' + spatial_graph_embedding_model_name).to(device)
            
#             if freeze_layer:
#                 #for child in spatial_graph_embedding_model.children():
#                     #for param in child.parameters():
#                         #param.requires_grad = False 
#                 for child in spatial_graph_embedding_model.graph_convolution_layers:
#                     for param in child.parameters():
#                         param.requires_grad = False 
                        
#             self.spatial_graph_embedding_models.append(spatial_graph_embedding_model)  
                                
#         print('spatial_graph_embedding_model')
#         print(self.spatial_graph_embedding_models[0])

#         # Fully-connected layer
#         self.fc1 = nn.Linear(hidden_spatial_graph_embedding_model, fc_hidden)
#         self.fc2 = nn.Linear(fc_hidden, n_class)
           
#     def forward(self, data, fold_id, mode):
#         x, adj = data[:2]
#         x_ori = x
#         random_walks = data[6]
#         random_walks = random_walks.reshape(random_walks.size()[0] * random_walks.size()[1], random_walks.size()[2], random_walks.size()[3])
        
#         # Node random walk model feature extractor
#         node_random_walk_model = self.node_random_walk_models[fold_id]
#         if mode == 'train':
#             node_random_walk_model.train()
#         else:
#             node_random_walk_model.eval()
#         LSTM_feature_extractor = torch.nn.Sequential(*list(node_random_walk_model.children())[:-2])
#         LSTM_feature_extractor_output = LSTM_feature_extractor(random_walks)
        
#         x = torch.mean(LSTM_feature_extractor_output[0], dim=1).squeeze()
#         x = x.reshape(adj.size()[0], adj.size()[1], x.size()[-1])
        
#         # Spatial graph embedding model feature extractor
#         if self.spatial_graph_embedding_model_name == 'GCN':
#             spatial_graph_embedding_model = self.spatial_graph_embedding_models[fold_id]
#             if mode == 'train':
#                 spatial_graph_embedding_model.train()
#             else:
#                 spatial_graph_embedding_model.eval()
#             for i in range(self.n_spatial_graph_embedding_model_layer):
#                 layers_feature_extractor = spatial_graph_embedding_model.graph_convolution_layers[i + 1]
#                 layers_feature_extractor_output = F.relu(layers_feature_extractor((x), (adj)))

#                 if i != self.n_spatial_graph_embedding_model_layer - 1:
#                     layers_feature_extractor_output = F.dropout(layers_feature_extractor_output, p=self.dropout, training=self.training)
        
#         # Readout
#         x = readout_function(layers_feature_extractor_output, self.readout)
                
#         # Linear combination ensemble layer
#         x = F.dropout(x, p=self.fc_dropout, training=self.training)
        
#         x = self.fc1(x)
#         x = self.fc2(x)
                    
#         return x

###

# import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from .GCN import GCN
# from .NodeRandomWalkNet import NodeRandomWalkNet
# from readouts.basic_readout import readout_function

# class ExpandedSpatialGraphEmbeddingNet(nn.Module):
#     def __init__(self, n_class, fc_hidden, dropout, device,
#         n_folds, dataset_name, readout,  
#         n_node_random_walk_model_layer = 1, hidden_node_random_walk_model = 64, n_spatial_graph_embedding_model_layer = 2, hidden_spatial_graph_embedding_model = 64,
#         spatial_graph_embedding_model_name = 'GCN', freeze_layer = True, fc_layer_type = 'A', concat_dropout = 0.2):
#         super(ExpandedSpatialGraphEmbeddingNet, self).__init__()
        
#         self.dropout = dropout
#         self.concat_dropout = concat_dropout
#         self.readout = readout
#         self.spatial_graph_embedding_model_name = spatial_graph_embedding_model_name
#         self.n_spatial_graph_embedding_model_layer = n_spatial_graph_embedding_model_layer
#         self.fc_layer_type = fc_layer_type
        
#         basic_path = './save_model/'
        
#         # Load node random walk model each fold
#         print('Load node random walk model ...')
        
#         self.node_random_walk_models = []
#         for i in range(n_folds):
#             node_random_walk_model_name = 'NodeRandomWalkNet_' + dataset_name + '_' + readout + '_' + str(i) + '_' + str(n_node_random_walk_model_layer) + '_h' + str(hidden_node_random_walk_model) + '.pt'
#             node_random_walk_model = torch.load(basic_path + 'NodeRandomWalkNet/' + node_random_walk_model_name).to(device)
            
#             if freeze_layer:
#                 for child in node_random_walk_model.children():
#                     for param in child.parameters():
#                         param.requires_grad = False
                
#             self.node_random_walk_models.append(node_random_walk_model)

#         print('node_random_walk_model')
#         print(self.node_random_walk_models[0])

#         # Load spatial graph embedding model each fold
#         print('Load spatial graph embedding model ...')
        
#         self.spatial_graph_embedding_models = []
#         for i in range(n_folds):
#             spatial_graph_embedding_model_name = self.spatial_graph_embedding_model_name + '_' + dataset_name + '_' + readout + '_' + str(i) + '_' + str(n_spatial_graph_embedding_model_layer) + '_h' + str(hidden_spatial_graph_embedding_model) + '.pt'
#             spatial_graph_embedding_model = torch.load(basic_path + self.spatial_graph_embedding_model_name + '/' + spatial_graph_embedding_model_name).to(device)
            
#             if freeze_layer:
#                 for child in spatial_graph_embedding_model.children():
#                     for param in child.parameters():
#                         param.requires_grad = False 
#                 for child in spatial_graph_embedding_model.graph_convolution_layers:
#                     for param in child.parameters():
#                         param.requires_grad = False 
                        
#             self.spatial_graph_embedding_models.append(spatial_graph_embedding_model)  
                                
#         print('spatial_graph_embedding_model')
#         print(self.spatial_graph_embedding_models[0])

#         # Fully-connected layer
#         if fc_layer_type == 'A':
#             self.fc1 = nn.Linear(fc_hidden * 2, int(fc_hidden / 4))
#             self.fc2 = nn.Linear(int(fc_hidden / 4), n_class)
#         elif fc_layer_type == 'B':
#             self.fc = nn.Linear(fc_hidden * 2, n_class)
#         elif fc_layer_type == 'C':
#             self.fc1 = nn.Linear(n_class * 2, 4 * n_class)
#             self.fc2 = nn.Linear(4 * n_class, n_class)
           
#     def forward(self, data, fold_id, mode):
#         x, adj = data[:2]
#         x_ori = x
#         random_walks = data[6]
#         random_walks = random_walks.reshape(random_walks.size()[0] * random_walks.size()[1], random_walks.size()[2], random_walks.size()[3])
        
#         # Node random walk model feature extractor
#         node_random_walk_model = self.node_random_walk_models[fold_id]
#         if mode == 'train':
#             node_random_walk_model.train()
#         else:
#             node_random_walk_model.eval()
#         LSTM_feature_extractor = torch.nn.Sequential(*list(node_random_walk_model.children())[:-2])
#         LSTM_feature_extractor_output = LSTM_feature_extractor(random_walks)
        
#         x = torch.mean(LSTM_feature_extractor_output[0], dim=1).squeeze()
#         x = x.reshape(adj.size()[0], adj.size()[1], x.size()[-1])
        
#         x = readout_function(x, self.readout)
        
#         if self.fc_layer_type == 'A' or self.fc_layer_type == 'B':
#             node_random_walk_model_feature_extractor = torch.nn.Sequential(*list(node_random_walk_model.children())[1:-1])
#         if self.fc_layer_type == 'C':
#             node_random_walk_model_feature_extractor = torch.nn.Sequential(*list(node_random_walk_model.children())[1:])
#         node_random_walk_model_feature_extractor_output = F.relu(node_random_walk_model_feature_extractor(x))
        
#         # Spatial graph embedding model feature extractor
#         if self.spatial_graph_embedding_model_name == 'GCN':
#             spatial_graph_embedding_model = self.spatial_graph_embedding_models[fold_id]
#             if mode == 'train':
#                 spatial_graph_embedding_model.train()
#             else:
#                 spatial_graph_embedding_model.eval()
#             for i in range(self.n_spatial_graph_embedding_model_layer):
#                 layers_feature_extractor = spatial_graph_embedding_model.graph_convolution_layers[i]
#                 if i == 0:
#                     layers_feature_extractor_output = F.relu(layers_feature_extractor((x_ori), (adj)))
#                 else:
#                     layers_feature_extractor_output = F.relu(layers_feature_extractor((layers_feature_extractor_output), (adj)))

#                 if i != self.n_spatial_graph_embedding_model_layer - 1:
#                     layers_feature_extractor_output = F.dropout(layers_feature_extractor_output, p=self.dropout, training=self.training)
                 
#             x = readout_function(layers_feature_extractor_output, self.readout)

#             if self.fc_layer_type == 'A' or self.fc_layer_type == 'B':    
#                 spatial_graph_embedding_model_feature_extractor = torch.nn.Sequential(*list(spatial_graph_embedding_model.children())[:-1])
#             if self.fc_layer_type == 'C':
#                 spatial_graph_embedding_model_feature_extractor = torch.nn.Sequential(*list(spatial_graph_embedding_model.children()))            
#             spatial_graph_embedding_model_feature_extractor_output = F.relu(spatial_graph_embedding_model_feature_extractor(x))
        
#         # Concat layer
#         if self.fc_layer_type == 'A' or self.fc_layer_type == 'B':    
#             concat = torch.cat((node_random_walk_model_feature_extractor_output, spatial_graph_embedding_model_feature_extractor_output), 1)
#         if self.fc_layer_type == 'C':    
#             concat = torch.cat((node_random_walk_model_feature_extractor_output, spatial_graph_embedding_model_feature_extractor_output), 1)
        
#         # Linear combination ensemble layer
#         x = F.dropout(concat, p=self.concat_dropout, training=self.training)
        
#         if self.fc_layer_type == 'A':
#             x = self.fc1(x)
#             x = self.fc2(x)
#         elif self.fc_layer_type == 'B':
#             x = self.fc(x)
#         elif self.fc_layer_type == 'C':
#             x = self.fc1(x)
#             x = self.fc2(x)
                    
#         return x