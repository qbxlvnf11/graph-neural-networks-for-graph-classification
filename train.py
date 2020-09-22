import argparse
from argparse import RawTextHelpFormatter
import sys
import numpy as np
import time
import statistics

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchsummary import summary

from datasets.graph_data_reader import DataReader, GraphData
from models.GCN import GCN
from models.GAT import GAT
from models.GraphSAGE import GraphSAGE
from models.APPNP import APPNP
from models.GIN import GIN
from models.GraphUNet import GraphUNet
from models.ARMA import ARMA
from models.SGCNN import SGCNN
from models.GraphResNet import GraphResNet
from models.GraphDenseNet import GraphDenseNet
from models.NodeRandomWalkNet import NodeRandomWalkNet
from models.ExpandedSpatialGraphEmbeddingNet import ExpandedSpatialGraphEmbeddingNet
from utils import create_directory, save_result_csv

model_list = ['GCN', 'GAT', 'GraphSAGE', 'APPNP', 'GIN', 'GraphUNet', 'ARMA', 'SGCNN', 'GraphResNet', 'GraphDenseNet', 'NodeRandomWalkNet', 'ExpandedSpatialGraphEmbeddingNet']
dataset_list = ['IMDB-BINARY', 'IMDB-MULTI', 'PROTEINS', 'ENZYMES', 'NCI1', 'MUTAG']
readout_list = ['max', 'avg', 'sum']

parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

# Target model & dataset & readout param
parser.add_argument('--model_list', nargs='+', required=True,
                    help='target train model list \n'+
                    'available model: ' + str(model_list)
                     + '\nable to choose multiple models \n'+
                     'ALL: choose all available model')
parser.add_argument('--dataset_list', nargs='+', required=True,
                    help='target graph classification dataset list \n'+
                    'available dataset: ' + str(dataset_list)
                     + '\nable to choose multiple datasets \n'+
                     'ALL: choose all available dataset')
parser.add_argument('--readout_list', nargs='+', required=True,
                    help='target readout method list \n'+
                    'available readout: ' + str(readout_list)
                     + '\nable to choose multiple readout methods \n'+
                     'ALL: choose all available readout')
                     
# Dataset param
parser.add_argument('--node_att', default='FALSE',
                    help='use additional float valued node attributes available in some datasets or not\n'+
                    'TRUE/FALSE')
parser.add_argument('--seed', type=int, default=111,
                    help='random seed')
parser.add_argument('--n_folds', type=int, default=10,
                    help='the number of folds in 10-cross validation')
parser.add_argument('--threads', type=int, default=0,
                    help='how many subprocesses to use for data loading \n'+
                    'default value 0 means that the data will be loaded in the main process')

# Graph data subsampling
parser.add_argument('--n_graph_subsampling', type=int, default=0,
                    help='the number of running graph subsampling each train graph data\n'+
                    'run subsampling 5 times: increasing graph data 5 times')
parser.add_argument('--graph_node_subsampling', default='TRUE',
                    help='TRUE: removing node randomly to subsampling and augmentation of graph dataset \n'+
                    'FALSE: removing edge randomly to subsampling and augmentation of graph dataset')
parser.add_argument('--graph_subsampling_rate', type=float, default=0.2,
                    help='graph subsampling rate')

# Learning param
parser.add_argument('--cuda', default='TRUE',
                    help='use cuda device in train process or not\n'+
                    'TRUE/FALSE')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size of data')                  
parser.add_argument('--epochs', type=int, default=50,
                    help='train epochs')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate of optimizer')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight decay of optimizer')
parser.add_argument('--log_interval', type=int, default=10,
                    help='print log interval in train process')
parser.add_argument('--save_model', default='TRUE',
                    help='save model or not\n'+
                    'TRUE/FALSE')

# Model param
parser.add_argument('--n_agg_layer', type=int, default=2,
                    help='the number of graph aggregation layers')
parser.add_argument('--agg_hidden', type=int, default=64,
                    help='size of hidden graph aggregation layer')
parser.add_argument('--fc_hidden', type=int, default=128,
                    help='size of fully-connected layer after readout')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout rate of layer')

# NodeRandomWalkNet param
parser.add_argument('--walk_length', type=int, default=20,
                    help='walk length of random walk')
parser.add_argument('--num_walk', type=int, default=10,
                    help='num_walk of random walk')
parser.add_argument('--p', type=float, default=0.65,
                    help='Possibility to return to the previous vertex, how well you navigate around')
parser.add_argument('--q', type=float, default=0.35,
                    help='Possibility of moving away from the previous vertex, how well you are exploring new places')

# ExpandedSpatialGraphEmbeddingNet param
parser.add_argument('--n_spatial_graph_embedding_model_layer', type=int, default=1,
                    help='the number of spatial graph embedding model layers')
parser.add_argument('--n_node_random_walk_model_layer', type=int, default=1,
                    help='the number of node random walk model layers')
parser.add_argument('--node_random_walk_model_name', default='LSTM',
                    help='node random walk model name\n'+
                    'LSTM/GRU')
#parser.add_argument('--freeze_layer', default='FALSE',
#                    help='the number of graph aggregation layers')
#parser.add_argument('--fc_layer_type', default='A',
#                    help='fully-connected layer type of ExpandedSpatialGraphEmbeddingNet')
#parser.add_argument('--concat_dropout', type=float, default=0.0,
#                    help='dropout rate of concat layer of ExpandedSpatialGraphEmbeddingNet')

args = parser.parse_args()

args.cuda = (args.cuda.upper()=='TRUE')
args.save_model = (args.save_model.upper()=='TRUE')
args.node_att = (args.node_att.upper()=='TRUE')
args.graph_node_subsampling = (args.graph_node_subsampling.upper()=='TRUE')
#args.freeze_layer = (args.freeze_layer.upper()=='TRUE')

# Build random walk or not (if mode == NodeRandomWalkNet or ExpandedSpatialGraphEmbeddingNet)
random_walk = ('NodeRandomWalkNet' in args.model_list or 'ExpandedSpatialGraphEmbeddingNet' in args.model_list or 'ALL' in args.model_list)

# Choose target graph classification model
if 'ALL' in args.model_list:
  args.model_list = model_list
else:
  for model in args.model_list:
    if not model in model_list:
      print('There are not available models in the target graph classification model list')
      sys.exit()

print('Target model list:', args.model_list)

# Choose target dataset
if 'ALL' in args.dataset_list:
  args.dataset_list = dataset_list
else:
  for dataset in args.dataset_list:
    if not dataset in dataset_list:
      print('There are not available datasets in the target graph dataset list')
      sys.exit()

print('Target dataset list:', args.dataset_list)

# Choose target readout
if 'ALL' in args.readout_list:
  args.readout_list = readout_list
else:
  for readout in args.readout_list:
    if not readout in readout_list:
      print('There are not available readouts in the target readout list')
      sys.exit()

print('Target readout list:', args.readout_list)

# Choose device
if args.cuda and torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'
  
print('Using device in train process:', device)

if args.n_graph_subsampling > 0 and args.graph_node_subsampling:
  print('graph subsampling: random node removal')
elif args.n_graph_subsampling > 0 and not args.graph_node_subsampling:
  print('graph subsampling: random edge removal')

for dataset_name in args.dataset_list:
    print('-'*50)
    
    print('Target dataset:', dataset_name)
    # Build graph data reader: IMDB-BINARY, IMDB-MULTI, ...
    datareader = DataReader(data_dir='./datasets/%s/' % dataset_name.upper(),
                        rnd_state=np.random.RandomState(args.seed),
                        folds=args.n_folds,           
                        use_cont_node_attr=False,
                        random_walk=random_walk,
                        num_walk=args.num_walk,
                        walk_length=args.walk_length,
                        p=args.p,
                        q=args.q,
                        node2vec_hidden=args.agg_hidden
                        )
    
    for model_name in args.model_list:
      for i, readout_name in enumerate(args.readout_list):
        print('-'*25)
        
        # Build graph classification model
        if model_name == 'GCN':
            model = GCN(n_feat=datareader.data['features_dim'],
                    n_class=datareader.data['n_classes'],
                    n_layer=args.n_agg_layer,
                    agg_hidden=args.agg_hidden,
                    fc_hidden=args.fc_hidden,
                    dropout=args.dropout,
                    readout=readout_name,
                    device=device).to(device)
        elif model_name == 'GAT':
            model = GAT(n_feat=datareader.data['features_dim'],
                    n_class=datareader.data['n_classes'],
                    n_layer=args.n_agg_layer,
                    agg_hidden=args.agg_hidden,
                    fc_hidden=args.fc_hidden,
                    dropout=args.dropout,
                    readout=readout_name,
                    device=device).to(device)
        elif model_name == 'GraphSAGE':
            model = GraphSAGE(n_feat=datareader.data['features_dim'],
                    n_class=datareader.data['n_classes'],
                    n_layer=args.n_agg_layer,
                    agg_hidden=args.agg_hidden,
                    fc_hidden=args.fc_hidden,
                    dropout=args.dropout,
                    readout=readout_name,
                    device=device).to(device)
        elif model_name == 'APPNP':
            model = APPNP(n_feat=datareader.data['features_dim'],
                    n_class=datareader.data['n_classes'],
                    n_layer=args.n_agg_layer,
                    agg_hidden=args.agg_hidden,
                    fc_hidden=args.fc_hidden,
                    dropout=args.dropout,
                    readout=readout_name,
                    device=device).to(device)
        elif model_name == 'GIN':
            model = GIN(n_feat=datareader.data['features_dim'],
                    n_class=datareader.data['n_classes'],
                    n_layer=args.n_agg_layer,
                    agg_hidden=args.agg_hidden,
                    fc_hidden=args.fc_hidden,
                    dropout=args.dropout,
                    readout=readout_name,
                    device=device).to(device)
        elif model_name == 'GraphUNet':
            model = GraphUNet(n_feat=datareader.data['features_dim'],
                    n_class=datareader.data['n_classes'],
                    n_layer=args.n_agg_layer,
                    agg_hidden=args.agg_hidden,
                    fc_hidden=args.fc_hidden,
                    dropout=args.dropout,
                    readout=readout_name,
                    device=device).to(device)
        elif model_name == 'GraphResNet':
            model = GraphResNet(n_feat=datareader.data['features_dim'],
                    n_class=datareader.data['n_classes'],
                    n_layer=args.n_agg_layer,
                    agg_hidden=args.agg_hidden,
                    fc_hidden=args.fc_hidden,
                    dropout=args.dropout,
                    readout=readout_name,
                    device=device).to(device)
        elif model_name == 'ARMA':
            model = ARMA(n_feat=datareader.data['features_dim'],
                    n_class=datareader.data['n_classes'],
                    n_layer=args.n_agg_layer,
                    agg_hidden=args.agg_hidden,
                    fc_hidden=args.fc_hidden,
                    dropout=args.dropout,
                    readout=readout_name,
                    device=device).to(device)
        elif model_name == 'SGCNN':
            model = SGCNN(n_feat=datareader.data['features_dim'],
                    n_class=datareader.data['n_classes'],
                    n_layer=args.n_agg_layer,
                    agg_hidden=args.agg_hidden,
                    fc_hidden=args.fc_hidden,
                    dropout=args.dropout,
                    readout=readout_name,
                    device=device).to(device)
        elif model_name == 'GraphDenseNet':
            model = GraphDenseNet(n_feat=datareader.data['features_dim'],
                    n_class=datareader.data['n_classes'],
                    n_layer=args.n_agg_layer,
                    agg_hidden=args.agg_hidden,
                    fc_hidden=args.fc_hidden,
                    dropout=args.dropout,
                    readout=readout_name,
                    device=device).to(device)
        elif model_name == 'NodeRandomWalkNet':
            model = NodeRandomWalkNet(n_feat=datareader.data['features_dim'],
                    n_class=datareader.data['n_classes'],
                    n_layer=args.n_agg_layer,
                    agg_hidden=args.agg_hidden,
                    fc_hidden=args.fc_hidden,
                    dropout=args.dropout,
                    readout=readout_name,
                    device=device,
                    walk_length=args.walk_length,
                    node_random_walk_model_name=args.node_random_walk_model_name).to(device)
        elif model_name == 'ExpandedSpatialGraphEmbeddingNet':
            model = ExpandedSpatialGraphEmbeddingNet(
                    n_class=datareader.data['n_classes'],
                    agg_hidden=args.agg_hidden,
                    fc_hidden=args.fc_hidden,
                    dropout=args.dropout,
                    readout=readout_name,
                    device=device,
                    walk_length=args.walk_length,
                    n_spatial_graph_embedding_model_layer=args.n_spatial_graph_embedding_model_layer,
                    n_node_random_walk_model_layer=args.n_node_random_walk_model_layer,
                    node_random_walk_model_name=args.node_random_walk_model_name).to(device) 
#        elif model_name == 'ExpandedSpatialGraphEmbeddingNet':
#            model = ExpandedSpatialGraphEmbeddingNet(
#                    n_class=datareader.data['n_classes'],
#                    fc_hidden=args.fc_hidden,
#                    dropout=args.dropout,
#                    readout=readout_name,
#                    device=device,
#                    dataset_name=dataset_name,
#                    n_folds=args.n_folds,
#                    freeze_layer=args.freeze_layer,
#                    fc_dropout=args.concat_dropout).to(device)  
#         elif model_name == 'ExpandedSpatialGraphEmbeddingNet':
#             model = ExpandedSpatialGraphEmbeddingNet(
#                     n_class=datareader.data['n_classes'],
#                     fc_hidden=args.fc_hidden,
#                     dropout=args.dropout,
#                     readout=readout_name,
#                     device=device,
#                     dataset_name=dataset_name,
#                     n_folds=args.n_folds,
#                     freeze_layer=args.freeze_layer,
#                     fc_layer_type=args.fc_layer_type,
#                     concat_dropout=args.concat_dropout).to(device)  
                                                         
        print(model)
        print('Readout:', readout_name)
        
        # Train & test each fold
        acc_folds = []
        time_folds = []
        for fold_id in range(args.n_folds):
            print('\nFOLD', fold_id)
            loaders = []
            for split in ['train', 'test']:
                # Build GDATA object
                if split == 'train':
                    gdata = GraphData(fold_id=fold_id,
                                       datareader=datareader,
                                       split=split,
                                       random_walk=random_walk,
                                       n_graph_subsampling=args.n_graph_subsampling,
                                       graph_node_subsampling=args.graph_node_subsampling,
                                       graph_subsampling_rate=args.graph_subsampling_rate)
                else:
                    gdata = GraphData(fold_id=fold_id,
                                       datareader=datareader,
                                       split=split,
                                       random_walk=random_walk,
                                       n_graph_subsampling=0,
                                       graph_node_subsampling=args.graph_node_subsampling,
                                       graph_subsampling_rate=args.graph_subsampling_rate)      
                
                # Build graph data pytorch loader
                loader = torch.utils.data.DataLoader(gdata, 
                                                     batch_size=args.batch_size,
                                                     shuffle=split.find('train') >= 0,
                                                     num_workers=args.threads,
                                                     drop_last=False)
                loaders.append(loader)
            
            # Total trainable param
            c = 0
            for p in filter(lambda p: p.requires_grad, model.parameters()):
                c += p.numel()
            print('N trainable parameters:', c)
            
            # Optimizer
            optimizer = optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=args.learning_rate,
                        weight_decay=args.weight_decay,
                        betas=(0.5, 0.999))
    
            scheduler = lr_scheduler.MultiStepLR(optimizer, [20, 30], gamma=0.1)
            
            # Train function
            def train(train_loader):
                total_time_iter = 0
                model.train()
                start = time.time()
                train_loss, n_samples = 0, 0
                for batch_idx, data in enumerate(train_loader):
                    for i in range(len(data)):
                        data[i] = data[i].to(device)
                    optimizer.zero_grad()
                    if model_name == 'ExpandedSpatialGraphEmbeddingNet':
                        output = model(data, fold_id, 'train')
                    else:
                        output = model(data)
                    loss = loss_fn(output, data[4])
                    loss.backward()
                    optimizer.step()
                    time_iter = time.time() - start
                    total_time_iter += time_iter
                    train_loss += loss.item() * len(output)
                    n_samples += len(output)
                    if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader) - 1:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f}) \tsec/iter: {:.4f}'.format(
                            epoch, n_samples, len(train_loader.dataset),
                            100. * (batch_idx + 1) / len(train_loader), loss.item(), train_loss / n_samples, time_iter / (batch_idx + 1) ))
                scheduler.step()
                return total_time_iter / (batch_idx + 1)
            
            # Test function
            def test(test_loader):
                print('Test model ...')
                model.eval()
                start = time.time()
                test_loss, correct, n_samples = 0, 0, 0
                for batch_idx, data in enumerate(test_loader):
                    for i in range(len(data)):
                        data[i] = data[i].to(device)
                    if model_name == 'ExpandedSpatialGraphEmbeddingNet':
                        output = model(data, fold_id, 'test')
                    else:
                        output = model(data)
                    loss = loss_fn(output, data[4], reduction='sum')
                    test_loss += loss.item()
                    n_samples += len(output)
                    pred = output.detach().cpu().max(1, keepdim=True)[1]
    
                    correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()
    
                time_iter = time.time() - start
    
                test_loss /= n_samples
    
                acc = 100. * correct / n_samples
                print('Test set (epoch {}): Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch, 
                                                                                                      test_loss, 
                                                                                                      correct, 
                                                                                                      n_samples, acc))
                return acc
            
            # Loss function
            loss_fn = F.cross_entropy
            
            total_time = 0
            for epoch in range(args.epochs):
                total_time_iter = train(loaders[0])
                total_time += total_time_iter
                acc = test(loaders[1])
            acc_folds.append(round(acc,2))
            time_folds.append(round(total_time/args.epochs,2))
            
            # Save model
            if args.save_model:
                print('Save model ...')
                create_directory('./save_model')
                create_directory('./save_model/' + model_name)
                
                if model_name == 'ExpandedSpatialGraphEmbeddingNet':
                    file_name = model_name + '_' + dataset_name + '_' + readout_name + '_' + str(fold_id) + '_' + args.node_random_walk_model_name + '_' + str(args.n_spatial_graph_embedding_model_layer) + '_' + str(args.n_node_random_walk_model_layer) + '_' + str(args.walk_length) + '_' + str(args.num_walk) + '_' + str(args.p) + '_' + str(args.q) + '_h' + str(args.agg_hidden) + '.pt'
                elif model_name == 'NodeRandomWalkNet':
                    file_name = model_name + '_' + dataset_name + '_' + readout_name + '_' + str(fold_id) + '_' + args.node_random_walk_model_name + '_' + str(args.n_agg_layer) + '_' + str(args.walk_length) + '_' + str(args.num_walk) + '_' + str(args.p) + '_' + str(args.q) + '_h' + str(args.agg_hidden) + '.pt'
                else:
                    file_name = model_name + '_' + dataset_name + '_' + readout_name + '_' + str(fold_id) + '_' + str(args.n_agg_layer) + '_h' + str(args.agg_hidden) + '.pt'
          
                torch.save(model, './save_model/' + model_name + '/' + file_name)
                print('Complete to save model')
    
        print(acc_folds)
        print('{}-fold cross validation avg acc (+- std): {} ({})'.format(args.n_folds, statistics.mean(acc_folds), statistics.stdev(acc_folds)))

        # Save 10-cross validation result as csv format
        create_directory('./test_result')
        create_directory('./test_result/' + model_name)
        
        result_list = []
        result_list.append(dataset_name)
        result_list.append(readout_name)
        for acc_fold in acc_folds:
          result_list.append(str(acc_fold))
        result_list.append(statistics.mean(acc_folds))
        result_list.append(statistics.stdev(acc_folds))
        result_list.append(statistics.mean(time_folds))
        
        if model_name == 'ExpandedSpatialGraphEmbeddingNet':
            file_name = model_name + '_' + args.node_random_walk_model_name + '_' + str(args.n_spatial_graph_embedding_model_layer) + '_' + str(args.n_node_random_walk_model_layer) + '_' + str(args.walk_length) + '_' + str(args.num_walk) + '_' + str(args.p) + '_' + str(args.q) + '_h' + str(args.agg_hidden) + '_' + '10_cross_validation.csv'
            if args.n_graph_subsampling > 0 and args.graph_node_subsampling:
              file_name = model_name + '_' + args.node_random_walk_model_name + '_' + str(args.n_spatial_graph_embedding_model_layer) + '_' + str(args.n_node_random_walk_model_layer) + '_' + str(args.walk_length) + '_' + str(args.num_walk) + '_' + str(args.p) + '_' + str(args.q) + '_h' + str(args.agg_hidden) + '_' + 'node_graph_subsampling' + '_' + '10_cross_validation.csv'
            elif args.n_graph_subsampling > 0:
              file_name = model_name + '_' + args.node_random_walk_model_name + '_' + str(args.n_spatial_graph_embedding_model_layer) + '_' + str(args.n_node_random_walk_model_layer) + '_' + str(args.walk_length) + '_' + str(args.num_walk) + '_' + str(args.p) + '_' + str(args.q) + '_h' + str(args.agg_hidden) + '_' + 'edge_graph_subsampling' + '_' + '10_cross_validation.csv'   
              
        elif model_name == 'NodeRandomWalkNet':
            file_name = model_name + '_' + args.node_random_walk_model_name + '_' + str(args.n_agg_layer) + '_' + str(args.walk_length) + '_' + str(args.num_walk) + '_' + str(args.p) + '_' + str(args.q) + '_h' + str(args.agg_hidden) + '_' + '10_cross_validation.csv'
            if args.n_graph_subsampling > 0 and args.graph_node_subsampling:
              file_name = model_name + '_' + args.node_random_walk_model_name + '_' + str(args.n_agg_layer) + '_' + str(args.walk_length) + '_' + str(args.num_walk) + '_' + str(args.p) + '_' + str(args.q) + '_h' + str(args.agg_hidden) + '_' + 'node_graph_subsampling' + '_' + '10_cross_validation.csv'
            elif args.n_graph_subsampling > 0:
              file_name = model_name + '_' + args.node_random_walk_model_name + '_' + str(args.n_agg_layer) + '_' + str(args.walk_length) + '_' + str(args.num_walk) + '_' + str(args.p) + '_' + str(args.q) + '_h' + str(args.agg_hidden) + '_' + 'edge_graph_subsampling' + '_' + '10_cross_validation.csv'            
                                                           
        else:
            file_name = model_name + '_' + str(args.n_agg_layer) + '_h' + str(args.agg_hidden) + '_' + '10_cross_validation.csv'
            if args.n_graph_subsampling > 0 and args.graph_node_subsampling:
              file_name = model_name + '_' + str(args.n_agg_layer) + '_h' + str(args.agg_hidden) + '_' + 'node_graph_subsampling' + '_' + '10_cross_validation.csv'
            elif args.n_graph_subsampling > 0:
              file_name = model_name + '_' + str(args.n_agg_layer) + '_h' + str(args.agg_hidden) + '_' + 'edge_graph_subsampling' + '_' + '10_cross_validation.csv'
        
        if i == 0:
          save_result_csv('./test_result/' + model_name + '/' + file_name, result_list, True)
        else:
          save_result_csv('./test_result/' + model_name + '/' + file_name, result_list, False)
        
        print('-'*25)
    print('-'*50)
