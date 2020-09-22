import networkx as nx
from node2vec import Node2Vec
import numpy as np

def get_node_random_walk(x_list, adj_list, node2vec_hidden, walk_length, num_walk, p, q, workers):
    node_random_walk_list = []
    
    print('walk_length:', walk_length)
    print('num_walk:', num_walk)
    print('p:', p)
    print('q:', q)
    
    for i, adj in enumerate(adj_list):
    
        if i % 15 == 0:
            print('node random walk ...', i, '/', len(adj_list))
            
        walk_dic = {}
        if type(adj).__module__ == np.__name__:
            G = nx.Graph(adj)
        else:
            G = nx.Graph(adj.to('cpu').numpy())
        
        node2vec = Node2Vec(graph=G, # The first positional argument has to be a networkx graph. Node names must be all integers or all strings. On the output model they will always be strings.
                    dimensions=node2vec_hidden, # Embedding dimensions (default: 128)
                    walk_length=walk_length, # number of nodes in each walks 
                    num_walks=num_walk, # Number of walks per node (default: 10)
                    p=p, # 전 꼭짓점으로 돌아올 가능성, 얼마나 주변을 잘 탐색하는가
                    q=q, # 전 꼭짓점으로부터 멀어질 가능성, 얼마나 새로운 곳을 잘 탐색하는가
                    weight_key=None, # On weighted graphs, this is the key for the weight attribute (default: 'weight')
                    workers=workers, # Number of workers for parallel execution (default: 1)
                    quiet = True
                   )

        # Dic key: target node number, dic value: random walks of target node
        for random_walk in node2vec.walks:
            if not int(random_walk[0]) in walk_dic:
                walk_dic[int(random_walk[0])] = []
            walk_dic[int(random_walk[0])].append(random_walk)
        
        # Get index of one value in one-hot vector
        if type(x_list[i]).__module__ == np.__name__:
            hot_index = np.where(x_list[i] == 1.0)[1]
        else:
            hot_index = np.where(x_list[i].to('cpu').numpy() == 1.0)[1]
         
        # Unify to Node Feature
        node_random_walk_list2 = []
        
        for a in range(len(adj)):
            walks = walk_dic[a]
            walks_list = []
            for walk in walks:
                walk2 = []
                for node in walk:
                    if not int(node) >= len(hot_index):
                        walk2.append(float(hot_index[int(node)]))
            
                # Padding and append
                walks_list.append([0.0] * (walk_length - len(walk2)) + walk2)

            node_random_walk_list2.append(np.array(walks_list))
        node_random_walk_list.append(np.array(node_random_walk_list2))
        
    return node_random_walk_list