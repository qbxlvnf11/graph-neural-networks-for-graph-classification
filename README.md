Description
=============

<div>
  <img src="https://user-images.githubusercontent.com/52263269/90756700-a6c67b80-e317-11ea-8c6f-d5ad2a77f03c.png" width="90%"></img>
</div>

#### - Graph Neural Networks (GNNs) for Graph Classification
  - Implementation of various neural graph classification model (not node classification)
  - Training and test of various Graph Neural Networks (GNNs) models using graph classification datasets
  - Input graph: graph adjacency matrix, graph node features matrix
  - Graph classification model (graph aggregating)
    - Get latent graph node featrue matrix
    - GCN, GAT, GIN, ...
  - Readout: transforming each latent node feature to one dimension vector for graph classification
  - Feature modeling: fully-connected layer

How to use
=============
#### - Details of parameter: references to help of argparse
```
python train.py --model_list GCN GAT --dataset_list ALL --readout_list ALL --n_agg_layer 2 -- agg_hidden 32
```


Contents
=============

#### - Available Model
  - Graph Convolutional Networks (GCNs): https://arxiv.org/abs/1609.02907
  - GraphSAGE: https://arxiv.org/pdf/1810.05997.pdf
  - Simple Graph Convolutional Networks (SGCNs): https://arxiv.org/pdf/1902.07153.pdf
  - Graph Attention Networks (GATs): https://arxiv.org/abs/1710.10903
  - Graph UNet: https://arxiv.org/pdf/1905.05178.pdf
  - Approximate Personalized Propagation of Neural Predictions (APPNP): https://arxiv.org/pdf/1810.05997.pdf
  - Graph Isomorphism Networks (GINs): https://arxiv.org/pdf/1810.00826.pdf
  - Graph Neural Networks with Convolutional ARMA Filters: https://arxiv.org/pdf/1901.01343.pdf
  - Graph ResNet
  - Graph DenseNet
  - Node RandomWalk Network
  - Expanded Spatial Graph Embedding Network
  
#### - Available Datasets
  - Node labels X, edge labels X: IMDB-BINARY, IMDB-MULTI
  - Node labels O, edge labels X: PROTEINS, ENZYMES, NCI1
  - Node labels O, edge labels O: MUTAG
  
#### - Available Readout
  - Basic readout: max, avg, sum
  
References
=============

#### - Graph classification data processing

https://github.com/bknyaz/graph_nn

Author
=============

#### - LinkedIn: https://www.linkedin.com/in/taeyong-kong-016bb2154

#### - Blog URL: https://blog.naver.com/qbxlvnf11

#### - Email: qbxlvnf11@google.com, qbxlvnf11@naver.com
