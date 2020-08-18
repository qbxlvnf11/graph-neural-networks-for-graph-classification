Description
=============

#### - Graph Neural Networks (GNNs) for Graph Classification
  - Implementation of various neural graph classification model (not node classification)
  - Training and test of various Graph Neural Networks (GNNs) models using graph classification datasets

How to use
=============
#### - Details of parameter: references to help of argparse
```
python train.py --model_list GCN GAT --dataset_list ALL --readout_list ALL --n_agg_layer 2 -- agg_hidden 32
```


Contents
=============

#### - Available Model
  - Graph Convolution Networks (GCNs): https://arxiv.org/abs/1609.02907
  - Graph Attention Networks (GATs): https://arxiv.org/abs/1710.10903
  - Graph Isomorphism Networks (GINs): https://arxiv.org/pdf/1810.00826.pdf
  - Graph UNet: https://arxiv.org/pdf/1905.05178.pdf
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
