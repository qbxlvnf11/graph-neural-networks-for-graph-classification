Description
=============

#### - Graph Neural Networks (GNNs) for Graph Classification
  - Implementation of various neural graph classification model (not node classification)
  - Training and test of various Graph Neural Networks (GNNs) models using graph classification datasets

How to use
=============

```
python main.py --load_data_path . --save_data_path .
```


Contents
=============

#### - Available Model
  - Graph Convolution Networks (GCNs): https://arxiv.org/abs/1609.02907
  - Graph Attention Networks (GATs): https://arxiv.org/abs/1710.10903
  
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

Contact
=============

#### - LinkedIn: https://www.linkedin.com/in/taeyong-kong-016bb2154

#### - Blog URL: https://blog.naver.com/qbxlvnf11

#### - Email: qbxlvnf11@google.com, qbxlvnf11@naver.com
