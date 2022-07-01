import sys
sys.path.append('/home/jxu8/Code/Explanability_bkd_gnn')
import torch
from torch import nn
from torch import device
import json
import os

#from Common.Utils.data_loader import load_data_fmnist, load_data_cifar10, load_data_mnist, load_data_tud_v2
from configs.config import args_parser
from util import inject_trigger


#from Common.Utils.data_split_iid import load_data_tud_split_v2
import numpy as np 
import torch.nn.functional as F
from GNN_common.data.data import LoadData
from GNN_common.nets.TUs_graph_classification.load_net import gnn_model  # import GNNs
import random
from torch.utils.data import DataLoader, random_split


def split_data(dataset):
    trainset, valset, testset = dataset.train[0], dataset.val[0], dataset.test[0]

    dataset_all = trainset + valset + testset
    total_size = len(dataset_all)
    count = 0
    for data in dataset_all:
        count += data[0].num_nodes()
    avg_nodes = count / total_size
    avg_nodes = round(avg_nodes)

    # resize the dataset into defined trainset and testset
    train_size = int(0.8*total_size)
    test_size = total_size - train_size
    length = [train_size, test_size]
    train_data, test_data = random_split(dataset_all, length)

    print('Training data: %d, Testing data: %d'%(train_size, test_size))
    return train_data, test_data, avg_nodes

def accuracy(scores, targets):
    scores = scores.detach().argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    return acc

def train(data_loader, model, optimizer, device):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        
        batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        loss = model.loss(batch_scores, batch_labels) 
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    
    return epoch_loss, epoch_train_acc

def test(data_loader, model, device):
    model.eval()
    epoch_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        
        batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        loss = model.loss(batch_scores, batch_labels) 
        epoch_loss += loss.detach().item()
        epoch_test_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_test_acc /= nb_data
    
    return epoch_loss, epoch_test_acc

def num_per_class(dataset, num_classes):
    for i in range(num_classes):
        tmp = torch.sum(dataset.all.graph_labels == i).item()
        print('class %d: %d'%(i, tmp))


if __name__ == '__main__':
    args = args_parser()
    with open(args.config) as f:
        config = json.load(f)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    raw_dir = './data'
    dataset = LoadData(config['dataset'], raw_dir)

    collate = dataset.collate
    

    MODEL_NAME = config['model']
    net_params = config['net_params']
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()
    params = config['params']
    net_params['in_dim'] = dataset.all.graph_lists[0].ndata['feat'][0].shape[0]
    num_classes = torch.max(dataset.all.graph_labels).item() + 1
    net_params['n_classes'] = num_classes
    #get number of each class
    num_per_class(dataset, num_classes)


    train_data, test_data, avg_nodes = split_data(dataset)
    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=params['lr_reduce_factor'],
                                                       patience=params['lr_schedule_patience'],
                                                       verbose=True)

    print("Model:\n{}".format(model))
    loss_func = nn.CrossEntropyLoss()
    # Load data
    drop_last = True if MODEL_NAME == 'DiffPool' else False
    dense = False
    
    # prepare backdoor training dataset and testing dataset
    train_trigger_graphs, test_trigger_graphs, trigger, final_idx = inject_trigger(train_data, test_data, avg_nodes, args)
    tmp_graphs = [train_data[idx] for idx in range(len(train_data)) if idx not in final_idx]

    bkd_train_dataset = train_trigger_graphs + train_data
    bkd_train_loader = DataLoader(bkd_train_dataset, batch_size=args.batch_size, shuffle=True,
                            drop_last=drop_last,
                            collate_fn=dataset.collate)
    bkd_attack_loader = DataLoader(test_trigger_graphs, batch_size=args.batch_size, shuffle=True,
                            drop_last=drop_last,
                            collate_fn=dataset.collate)
    clean_test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True,
                            drop_last=drop_last,
                            collate_fn=dataset.collate)
    #import pdb
    #pdb.set_trace()
    acc_record = [0]
    counts = 0
    #for epoch in range(config.num_epochs):
    
    for epoch in range(params['epochs']):
        print('epoch:',epoch)
        train_loss, train_acc = train(bkd_train_loader, model, optimizer, device)
        att_loss, asr = test(bkd_attack_loader, model, device)
        test_loss, test_acc = test(clean_test_loader, model, device)
        print('Train loss: %.3f, Train acc: %.3f, Test loss: %.3f, Test acc: %.3f'%(train_loss, train_acc, test_loss, test_acc))
        print('Attack loss: %.3f, Attack success rate: %.3f'%(att_loss, asr))

        if not args.filename == "":
            save_path = os.path.join(args.filename, '%d'%args.seed, config['model'] + '_' + net_params['readout'] + '_' + config['dataset'] + '_%d_%.2f_%.2f.txt'\
                %(args.bkd_size, args.poisoning_intensity, args.density))
            path = os.path.split(save_path)[0]
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
            with open(save_path, 'a') as f:
                f.write('%.3f %.3f %.3f %.3f %.3f '%(train_loss, train_acc, test_loss, test_acc, asr))
                f.write('\n')
                
                
