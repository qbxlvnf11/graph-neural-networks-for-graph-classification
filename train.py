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
from utils import create_directory, save_result_csv

model_list = ['GCN', 'GAT']
dataset_list = ['IMDB-BINARY', 'IMDB-MULTI', 'PROTEINS', 'ENZYMES', 'NCI1', 'MUTAG']
readout_list = ['max', 'avg', 'sum']

parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

# Target model & dataset & readout param
parser.add_argument('--model_list', nargs='+', required=True,
                    help='target train model list \navailable model: ' + str(model_list)
                     + '\nable to get multiple values\nALL: choose all available model')
parser.add_argument('--dataset_list', nargs='+', required=True,
                    help='target graph classification dataset list \navailable dataset: ' + str(dataset_list)
                     + '\nable to get multiple values\nALL: choose all available dataset')
parser.add_argument('--readout_list', nargs='+', required=True,
                    help='target readout method list \navailable readout: ' + str(readout_list)
                     + '\nable to get multiple values\nALL: choose all available readout')
                     
# Dataset param
parser.add_argument('--node_att', default='FALSE',
                    help='use additional float valued node attributes available in some datasets or not\nTRUE/FALSE')
parser.add_argument('--seed', type=int, default=111,
                    help='random seed')
parser.add_argument('--n_folds', type=int, default=10,
                    help='the number of folds in 10-cross validation')
parser.add_argument('--threads', type=int, default=0,
                    help='how many subprocesses to use for data loading \ndefault value 0 means that the data will be loaded in the main process')
# Graph data subsampling
parser.add_argument('--n_graph_subsampling', type=int, default=0,
                    help='the number of running graph subsampling each train graph data\nrun subsampling 5 times: increasing graph data 5 times')
parser.add_argument('--graph_node_subsampling', default='TRUE',
                    help='TRUE: removing node randomly to subsampling and augmentation of graph dataset'
                    + '\nFALSE: removing edge randomly to subsampling and augmentation of graph dataset')
parser.add_argument('--graph_subsampling_rate', type=float, default=0.2,
                    help='graph subsampling rate')

# Learning param
parser.add_argument('--cuda', default='TRUE',
                    help='use cuda device in train process or not\nTRUE/FALSE')
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
                    help='save model or not\nTRUE/FALSE')

# Model param
parser.add_argument('--n_agg_layer', type=int, default=2,
                    help='the number of graph aggregation layers')
parser.add_argument('--agg_hidden', type=int, default=32,
                    help='size of hidden graph aggregation layer')
parser.add_argument('--fc_hidden', type=int, default=128,
                    help='size of fully-connected layer after readout')
parser.add_argument('--dropout', type=float, default=0.25,
                    help='dropout rate of layer')
                    
args = parser.parse_args()

args.cuda = (args.cuda.upper()=='TRUE')
args.save_model = (args.save_model.upper()=='TRUE')
args.node_att = (args.node_att.upper()=='TRUE')
args.graph_node_subsampling = (args.graph_node_subsampling.upper()=='TRUE')

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

for dataset_name in args.dataset_list:
    print('-'*50)
    
    print('Target dataset:', dataset_name)
    # Build graph data reader: IMDB-BINARY, IMDB-MULTI, ...
    datareader = DataReader(data_dir='./datasets/%s/' % dataset_name.upper(),
                        rnd_state=np.random.RandomState(args.seed),
                        folds=args.n_folds,                    
                        use_cont_node_attr=False)
    
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

        print(model)
        print('Readout:', readout_name)
        
        # Train & test each fold
        acc_folds = []
        for fold_id in range(args.n_folds):
            print('\nFOLD', fold_id)
            loaders = []
            for split in ['train', 'test']:
                # Build GDATA object
                if split == 'train':
                    gdata = GraphData(fold_id=fold_id,
                                       datareader=datareader,
                                       split=split,
                                       n_graph_subsampling=args.n_graph_subsampling,
                                       graph_node_subsampling=args.graph_node_subsampling,
                                       graph_subsampling_rate=args.graph_subsampling_rate)
                else:
                    gdata = GraphData(fold_id=fold_id,
                                       datareader=datareader,
                                       split=split,
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
                model.train()
                start = time.time()
                train_loss, n_samples = 0, 0
                for batch_idx, data in enumerate(train_loader):
                    for i in range(len(data)):
                        data[i] = data[i].to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = loss_fn(output, data[4])
                    loss.backward()
                    optimizer.step()
                    time_iter = time.time() - start
                    train_loss += loss.item() * len(output)
                    n_samples += len(output)
                    if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader) - 1:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f}) \tsec/iter: {:.4f}'.format(
                            epoch, n_samples, len(train_loader.dataset),
                            100. * (batch_idx + 1) / len(train_loader), loss.item(), train_loss / n_samples, time_iter / (batch_idx + 1) ))
                scheduler.step()
            
            # Test function
            def test(test_loader):
                print('Test model ...')
                model.eval()
                start = time.time()
                test_loss, correct, n_samples = 0, 0, 0
                for batch_idx, data in enumerate(test_loader):
                    for i in range(len(data)):
                        data[i] = data[i].to(device)
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
    
            loss_fn = F.cross_entropy
            for epoch in range(args.epochs):
                train(loaders[0])
                acc = test(loaders[1])
            # Save model
            if args.save_model:
                print('Save model ...')
                create_directory('./save_model')
                
                file_name = model_name + '_' + dataset_name + '_' + readout_name + '_' + str(fold_id) + '.pt'
          
                torch.save(model, './save_model/' + file_name)
                print('Complete to save model')
            acc_folds.append(round(acc,2))
    
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
        
        file_name = model_name + '_' + str(args.n_agg_layer) + '_' + '10_cross_validation.csv'
        if args.n_graph_subsampling > 0 and args.graph_node_subsampling:
          file_name = model_name + '_' + str(args.n_agg_layer) + '_' + 'node_graph_subsampling' + '_' + '10_cross_validation.csv'
        elif args.n_graph_subsampling > 0:
          file_name = model_name + '_' + str(args.n_agg_layer) + '_' + 'edge_graph_subsampling' + '_' + '10_cross_validation.csv'
        
        if i == 0:
          save_result_csv('./test_result/' + model_name + '/' + file_name, result_list, True)
        else:
          save_result_csv('./test_result/' + model_name + '/' + file_name, result_list, False)
        
        print('-'*25)
    print('-'*50)
