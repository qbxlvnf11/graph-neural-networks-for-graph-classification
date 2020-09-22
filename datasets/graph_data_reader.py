import networkx as nx
import numpy as np
import os
from os.path import join as pjoin
import copy

import torch

from .graph_subsampling import graph_dataset_subsampling
from .graph_node_random_walk import get_node_random_walk

"""
References: https://github.com/bknyaz/graph_nn
"""

class DataReader():

    '''
    Class to read the txt files containing all data of the dataset
    '''
    def __init__(self,
                 data_dir,  # Folder with txt files
                 random_walk,
                 node2vec_hidden,
                 walk_length,
                 num_walk,
                 p,
                 q,
                 workers=3,
                 rnd_state=None,
                 use_cont_node_attr=False,  # Use or not additional float valued node attributes available in some datasets
                 folds=10):

        self.data_dir = data_dir
        self.rnd_state = np.random.RandomState() if rnd_state is None else rnd_state
        self.use_cont_node_attr = use_cont_node_attr
        files = os.listdir(self.data_dir)
        
        print('data path:', self.data_dir)
        
        data = {}
        
        # Read adj list
        nodes, graphs = self.read_graph_nodes_relations(list(filter(lambda f: f.find('graph_indicator') >= 0, files))[0]) 
        data['adj_list'] = self.read_graph_adj(list(filter(lambda f: f.find('_A') >= 0, files))[0], nodes, graphs)        
                
        print('complete to build adjacency matrix list')
        
        # Make node count list
        data['node_count_list'] = self.get_node_count_list(data['adj_list'])
        
        print('complete to build node count list')

        # Make edge matrix list
        data['edge_matrix_list'], data['max_edge_matrix'] = self.get_edge_matrix_list(data['adj_list'])
        
        print('complete to build edge matrix list')

        # Make node count list
        data['edge_matrix_count_list'] = self.get_edge_matrix_count_list(data['edge_matrix_list'])
        
        print('complete to build edge matrix count list')
        
        # Make degree_features and max neighbor list
        degree_features = self.get_node_features_degree(data['adj_list'])
        data['max_neighbor_list'] = self.get_max_neighbor(degree_features)
        
        print('complete to build max neighbor list')
       
        # Read features or make features
        if len(list(filter(lambda f: f.find('node_labels') >= 0, files))) != 0:
            print('node label: node label in dataset')
            data['features'] = self.read_node_features(list(filter(lambda f: f.find('node_labels') >= 0, files))[0], 
                                                     nodes, graphs, fn=lambda s: int(s.strip()))
        else:
            print('node label: degree of nodes')
            data['features'] = degree_features
            
        print('complete to build node features list')
        
        data['targets'] = np.array(self.parse_txt_file(list(filter(lambda f: f.find('graph_labels') >= 0, files))[0],
                                                       line_parse_fn=lambda s: int(float(s.strip()))))
                                                       
        print('complete to build targets list')
        
        if self.use_cont_node_attr:
            data['attr'] = self.read_node_features(list(filter(lambda f: f.find('node_attributes') >= 0, files))[0], 
                                                   nodes, graphs, fn=lambda s: np.array(list(map(float, s.strip().split(',')))))
        
        features, n_edges, degrees = [], [], []
        for sample_id, adj in enumerate(data['adj_list']):
            N = len(adj) # Number of nodes
            if data['features'] is not None:
                assert N == len(data['features'][sample_id]), (N, len(data['features'][sample_id]))
            n = np.sum(adj) # Total sum of edges
            n_edges.append( int(n / 2) ) # Undirected edges, so need to divide by 2
            if not np.allclose(adj, adj.T):
                print(sample_id, 'not symmetric')
            degrees.extend(list(np.sum(adj, 1)))
            features.append(np.array(data['features'][sample_id]))
                        
        # Create features over graphs as one-hot vectors for each node
        features_all = np.concatenate(features)
        features_min = features_all.min()
        features_dim = int(features_all.max() - features_min + 1) # Number of possible values
        
        features_onehot = []
        for i, x in enumerate(features):
            feature_onehot = np.zeros((len(x), features_dim))
            for node, value in enumerate(x):
                feature_onehot[node, value - features_min] = 1
            if self.use_cont_node_attr:
                feature_onehot = np.concatenate((feature_onehot, np.array(data['attr'][i])), axis=1)
            features_onehot.append(feature_onehot)

        if self.use_cont_node_attr:
            features_dim = features_onehot[0].shape[1]
            
        shapes = [len(adj) for adj in data['adj_list']]
        labels = data['targets'] # Graph class labels
        labels -= np.min(labels) # To start from 0
        N_nodes_max = np.max(shapes)

        classes = np.unique(labels)
        n_classes = len(classes)

        if not np.all(np.diff(classes) == 1):
            print('making labels sequential, otherwise pytorch might crash')
            labels_new = np.zeros(labels.shape, dtype=labels.dtype) - 1
            for lbl in range(n_classes):
                labels_new[labels == classes[lbl]] = lbl
            labels = labels_new
            classes = np.unique(labels)
            assert len(np.unique(labels)) == n_classes, np.unique(labels)
        
        print('-'*50)
        print('The number of graphs:', len(data['adj_list']))
        print('N nodes avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(shapes), np.std(shapes), np.min(shapes), np.max(shapes)))
        print('N edges avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(n_edges), np.std(n_edges), np.min(n_edges), np.max(n_edges)))
        print('Node degree avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(degrees), np.std(degrees), np.min(degrees), np.max(degrees)))
        print('Node features dim: \t\t%d' % features_dim)
        print('N classes: \t\t\t%d' % n_classes)
        print('Classes: \t\t\t%s' % str(classes))
        for lbl in classes:
            print('Class %d: \t\t\t%d samples' % (lbl, np.sum(labels == lbl)))

        for u in np.unique(features_all):
            print('feature {}, count {}/{}'.format(u, np.count_nonzero(features_all == u), len(features_all)))
        
        N_graphs = len(labels)  # Number of samples (graphs) in data
        assert N_graphs == len(data['adj_list']) == len(features_onehot), 'invalid data'

        # Create test sets first
        train_ids, test_ids = self.split_ids(np.arange(N_graphs), rnd_state=self.rnd_state, folds=folds)

        # Create train sets
        splits = []
        for fold in range(folds):
            splits.append({'train': train_ids[fold],
                           'test': test_ids[fold]})

        data['features_onehot'] = features_onehot
        data['targets'] = labels
        data['splits'] = splits
        data['N_nodes_max'] = np.max(shapes)  # Max number of nodes
        data['features_dim'] = features_dim
        data['n_classes'] = n_classes

        # Make neighbor dictionary
        #data['neighbor_dic_list'] = self.get_neighbor_dic_list(data['adj_list'], data['N_nodes_max'])
        
        #print('complete to build neighbor dictionary list')
        
        # Make node randomwalk
        if random_walk:
            print('building node randomwalk list ...')
            data['random_walks'] = get_node_random_walk(data['features_onehot'], data['adj_list'], node2vec_hidden, walk_length, num_walk, p, q, workers)
            print('complete to build node randomwalk list')
        
        self.data = data

    def split_ids(self, ids_all, rnd_state=None, folds=10):
        n = len(ids_all)
        ids = ids_all[rnd_state.permutation(n)]
        stride = int(np.ceil(n / float(folds)))
        test_ids = [ids[i: i + stride] for i in range(0, n, stride)]
        assert np.all(np.unique(np.concatenate(test_ids)) == sorted(ids_all)), 'some graphs are missing in the test sets'
        assert len(test_ids) == folds, 'invalid test sets'
        train_ids = []
        for fold in range(folds):
            train_ids.append(np.array([e for e in ids if e not in test_ids[fold]]))
            assert len(train_ids[fold]) + len(test_ids[fold]) == len(np.unique(list(train_ids[fold]) + list(test_ids[fold]))) == n, 'invalid splits'

        return train_ids, test_ids

    def parse_txt_file(self, fpath, line_parse_fn=None):
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()
        data = [line_parse_fn(s) if line_parse_fn is not None else s for s in lines]
        
        return data
    
    def read_graph_adj(self, fpath, nodes, graphs):
        edges = self.parse_txt_file(fpath, line_parse_fn=lambda s: s.split(','))
        adj_dict = {}
        for edge in edges:
            node1 = int(edge[0].strip()) - 1  # -1 because of zero-indexing in our code
            node2 = int(edge[1].strip()) - 1
            graph_id = nodes[node1]
            assert graph_id == nodes[node2], ('invalid data', graph_id, nodes[node2])
            if graph_id not in adj_dict:
                n = len(graphs[graph_id])
                adj_dict[graph_id] = np.zeros((n, n))
            ind1 = np.where(graphs[graph_id] == node1)[0]
            ind2 = np.where(graphs[graph_id] == node2)[0]
            assert len(ind1) == len(ind2) == 1, (ind1, ind2)
            adj_dict[graph_id][ind1, ind2] = 1
            
        adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]
        
        return adj_list
        
    def read_graph_nodes_relations(self, fpath):
        graph_ids = self.parse_txt_file(fpath, line_parse_fn=lambda s: int(s.rstrip()))
        nodes, graphs = {}, {}
        for node_id, graph_id in enumerate(graph_ids):
            if graph_id not in graphs:
                graphs[graph_id] = []
            graphs[graph_id].append(node_id)
            nodes[node_id] = graph_id
        graph_ids = np.unique(list(graphs.keys()))
        for graph_id in graphs:
            graphs[graph_id] = np.array(graphs[graph_id])
        return nodes, graphs

    def read_node_features(self, fpath, nodes, graphs, fn):
        node_features_all = self.parse_txt_file(fpath, line_parse_fn=fn)
        node_features = {}
        
        for node_id, x in enumerate(node_features_all):
            graph_id = nodes[node_id]
            if graph_id not in node_features:
                node_features[graph_id] = [ None ] * len(graphs[graph_id])
            ind = np.where(graphs[graph_id] == node_id)[0]
            assert len(ind) == 1, ind
            assert node_features[graph_id][ind[0]] is None, node_features[graph_id][ind[0]]
            node_features[graph_id][ind[0]] = x
        node_features_lst = [node_features[graph_id] for graph_id in sorted(list(graphs.keys()))]
        return node_features_lst

    def get_node_features_degree(self, adj_list):
        node_features_list = []

        for adj in adj_list:
            sub_list = []
            for feature in nx.from_numpy_matrix(np.array(adj)).degree():
                sub_list.append(feature[1])
            node_features_list.append(np.array(sub_list))

        return node_features_list
        
    def get_max_neighbor(self, degree_list):
        max_neighbor_list = []
        
        for degrees in degree_list:
            max_neighbor_list.append(int(max(degrees)))

        return max_neighbor_list

    def get_node_count_list(self, adj_list):
        node_count_list = []
        
        for adj in adj_list:
            node_count_list.append(len(adj))
                        
        return node_count_list

    def get_edge_matrix_list(self, adj_list):
        edge_matrix_list = []
        max_edge_matrix = 0
        
        for adj in adj_list:
            edge_matrix = []
            for i in range(len(adj)):
                for j in range(len(adj[0])):
                    if adj[i][j] == 1:
                        edge_matrix.append((i,j))
            if len(edge_matrix) > max_edge_matrix:
                max_edge_matrix = len(edge_matrix)
            edge_matrix_list.append(np.array(edge_matrix))
                        
        return edge_matrix_list, max_edge_matrix

    def get_edge_matrix_count_list(self, edge_matrix_list):
        edge_matrix_count_list = []
        
        for edge_matrix in edge_matrix_list:
            edge_matrix_count_list.append(len(edge_matrix))
                        
        return edge_matrix_count_list
    
    def get_neighbor_dic_list(self, adj_list, N_nodes_max):
        neighbor_dic_list = []
        
        for adj in adj_list:
            neighbors = []
            for i, row in enumerate(adj):
                idx = np.where(row == 1.0)[0].tolist()
                idx = np.pad(idx, (0, N_nodes_max - len(idx)), 'constant', constant_values=0)
                neighbors.append(idx)
            for a in range(i, N_nodes_max - 1):
                neighbors.append(np.array([0]*136))
            neighbor_dic_list.append(np.array(neighbors))
        
        return neighbor_dic_list    

class GraphData(torch.utils.data.Dataset):
    def __init__(self,
                 fold_id,
                 datareader,
                 split,
                 random_walk,
                 n_graph_subsampling,
                 graph_node_subsampling,
                 graph_subsampling_rate):
        
        self.random_walk = random_walk
        
        self.set_fold(datareader.data, split, fold_id, n_graph_subsampling, graph_node_subsampling, graph_subsampling_rate)

    def set_fold(self, data, split, fold_id, n_graph_subsampling, graph_node_subsampling, graph_subsampling_rate):
        self.total = len(data['targets'])
        self.N_nodes_max = data['N_nodes_max']
        self.max_edge_matrix = data['max_edge_matrix']
        self.n_classes = data['n_classes']
        self.features_dim = data['features_dim']
        self.idx = data['splits'][fold_id][split]
        
        # Use deepcopy to make sure we don't alter objects in folds
        self.labels = copy.deepcopy([data['targets'][i] for i in self.idx])
        self.adj_list = copy.deepcopy([data['adj_list'][i] for i in self.idx])
        self.features_onehot = copy.deepcopy([data['features_onehot'][i] for i in self.idx])
        self.max_neighbor_list = copy.deepcopy([data['max_neighbor_list'][i] for i in self.idx])
        self.edge_matrix_list = copy.deepcopy([data['edge_matrix_list'][i] for i in self.idx])
        self.node_count_list = copy.deepcopy([data['node_count_list'][i] for i in self.idx])
        self.edge_matrix_count_list = copy.deepcopy([data['edge_matrix_count_list'][i] for i in self.idx])
        #self.neighbor_dic_list = copy.deepcopy([data['neighbor_dic_list'][i] for i in self.idx])
        
        if self.random_walk:
            self.random_walks = copy.deepcopy([data['random_walks'][i] for i in self.idx])
        
        if n_graph_subsampling:
            self.adj_list, self.features_onehot, self.labels, self.max_neighbor_list, self.neighbor_dic_list = graph_dataset_subsampling(self.adj_list,
                                                                                   self.features_onehot, 
                                                                                   self.labels,
                                                                                   self.max_neighbor_list,
                                                                                   rate=graph_subsampling_rate,
                                                                                   repeat_count=n_graph_subsampling,
                                                                                   node_removal=graph_node_subsampling,
                                                                                   log=False)
        
        print('%s: %d/%d' % (split.upper(), len(self.labels), len(data['targets'])))
        
        # Sample indices for this epoch
        if n_graph_subsampling:
            self.indices = np.arange(len(self.idx) * n_graph_subsampling)  
        else:
            self.indices = np.arange(len(self.idx))
        
    def pad(self, mtx, desired_dim1, desired_dim2=None, value=0, mode='edge_matrix'):
        sz = mtx.shape
        #assert len(sz) == 2, ('only 2d arrays are supported', sz)
        
        if len(sz) == 2:
            if desired_dim2 is not None:
                  mtx = np.pad(mtx, ((0, desired_dim1 - sz[0]), (0, desired_dim2 - sz[1])), 'constant', constant_values=value)
            else:
                  mtx = np.pad(mtx, ((0, desired_dim1 - sz[0]), (0, 0)), 'constant', constant_values=value)
        elif len(sz) == 3:
            mtx = np.pad(mtx, ((0, desired_dim1 - sz[0]), (0, 0), (0, 0)), 'constant', constant_values=value)
        
        return mtx
    
    def nested_list_to_torch(self, data):
        #if isinstance(data, dict):
            #keys = list(data.keys())           
        for i in range(len(data)):
            #if isinstance(data, dict):
                #i = keys[i]
            if isinstance(data[i], np.ndarray):
                data[i] = torch.from_numpy(data[i]).float()
            #elif isinstance(data[i], list):
                #data[i] = list_to_torch(data[i])
        return data
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        index = self.indices[index]
        N_nodes_max = self.N_nodes_max
        N_nodes = self.adj_list[index].shape[0]
        graph_support = np.zeros(self.N_nodes_max)
        graph_support[:N_nodes] = 1
        
        if self.random_walk:
            return self.nested_list_to_torch([self.pad(self.features_onehot[index].copy(), self.N_nodes_max),  # Node_features
                                          self.pad(self.adj_list[index], self.N_nodes_max, self.N_nodes_max),  # Adjacency matrix
                                          graph_support,  # Mask with values of 0 for dummy (zero padded) nodes, otherwise 1 
                                          N_nodes,
                                          int(self.labels[index]),
                                          int(self.max_neighbor_list[index]),
                                          self.pad(self.random_walks[index])])                           
        else:
            return self.nested_list_to_torch([self.pad(self.features_onehot[index].copy(), self.N_nodes_max),  # Node_features
                                          self.pad(self.adj_list[index], self.N_nodes_max, self.N_nodes_max),  # Adjacency matrix
                                          graph_support,  # Mask with values of 0 for dummy (zero padded) nodes, otherwise 1 
                                          N_nodes,
                                          int(self.labels[index]),
                                          int(self.max_neighbor_list[index]),
                                          self.pad(self.edge_matrix_list[index], self.max_edge_matrix),
                                          int(self.node_count_list[index]),
                                          int(self.edge_matrix_count_list[index])])            