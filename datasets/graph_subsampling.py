import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import copy

def graph_subsampling_random_node_removal(adj, rate, log=True):
    if log: graph_log('--input graph--', adj)

    # Get number of the node
    node_count = len(adj)
    
    # Choose removed node
    remove_node_list = random.sample(range(0, node_count), int(node_count * rate))
    remove_node_list.sort()
    
    if log: print('remove node list:', remove_node_list)
    
    # Process 1: remove a node and connected edge
    subsampling_graph_adj = copy.deepcopy(adj)
    
    subsampling_graph_adj = np.delete(subsampling_graph_adj, remove_node_list, axis=1)
    subsampling_graph_adj = np.delete(subsampling_graph_adj, remove_node_list, axis=0)
    
    if log: graph_log('--subsampling graph--', subsampling_graph_adj)
        
    # Process 2: remove a node without a connected edge
    subsampling_graph_node_count = len(subsampling_graph_adj)
    node_without_connected_edge_list = []

    for i in range(subsampling_graph_node_count):
        if sum(subsampling_graph_adj[i]) == subsampling_graph_adj[i][i]:
            node_without_connected_edge_list.append(i)
                
    subsampling_graph_adj = np.delete(subsampling_graph_adj, node_without_connected_edge_list, axis=1)
    subsampling_graph_adj = np.delete(subsampling_graph_adj, node_without_connected_edge_list, axis=0)
    
    if log: graph_log('--subsampling graph--', subsampling_graph_adj)
        
    return subsampling_graph_adj
    

def graph_subsampling_random_edge_removal(adj, rate, log=True):
    if log: graph_log('--input graph--', adj)
        
    # Get number of the edge except for self roop
    node_count = len(adj)
    edge_count = 0
    for i in range(node_count):
        for a in range(node_count):
            if (i < a) and (adj[i][a] > 0):
                edge_count += 1
                    
    # Choose removed edge
    remove_edge_list = random.sample(range(0, edge_count), int(edge_count * rate))
    remove_edge_list.sort()
    
    if log: print('remove edge list:', remove_edge_list)
    
    # Remove edge
    subsampling_graph_adj = copy.deepcopy(adj)
    count = 0
    
    for i in range(node_count):
        for a in range(node_count):
            if (i < a) and (subsampling_graph_adj[i][a] > 0):   
                if count in remove_edge_list:
                    subsampling_graph_adj[i][a] = 0
                    subsampling_graph_adj[a][i] = 0
                count +=1
                
    if log: graph_log('--subsampling graph--', subsampling_graph_adj)
        
    # Remove a node without a connected edge
    subsampling_graph_node_count = len(subsampling_graph_adj)
    node_without_connected_edge_list = []

    for i in range(subsampling_graph_node_count):
        if sum(subsampling_graph_adj[i]) == subsampling_graph_adj[i][i]:
            node_without_connected_edge_list.append(i)
                
    subsampling_graph_adj = np.delete(subsampling_graph_adj, node_without_connected_edge_list, axis=1)
    subsampling_graph_adj = np.delete(subsampling_graph_adj, node_without_connected_edge_list, axis=0)
    
    if log: graph_log('--subsampling graph--', subsampling_graph_adj)
        
    return subsampling_graph_adj

def graph_dataset_subsampling(adj_list, node_features_list, label_list, max_neighbor_list, rate, repeat_count, node_removal=True, log=True):
    
    subsampling_adj_list = []
    subsampling_node_features_list = []
    subsampling_label_list = []
    subsampling_max_neighbor_list = []
    
    for i in range(len(adj_list)):
        for a in range(repeat_count):
            if node_removal:
                subsampling_adj_list.append(graph_subsampling_random_node_removal(adj_list[i], rate, log))
            else:
                subsampling_adj_list.append(graph_subsampling_random_edge_removal(adj_list[i], rate, log))
            
            subsampling_node_features_list.append(node_features_list[i])
            subsampling_label_list.append(label_list[i])
            subsampling_max_neighbor_list.append(max_neighbor_list[i])

    return np.array(subsampling_adj_list), np.array(subsampling_node_features_list), np.array(subsampling_label_list), subsampling_max_neighbor_list