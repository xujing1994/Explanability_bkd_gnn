import sys
sys.path.append('/home/jxu8/Code/Explanability_bkd_gnn')
#print(sys.path)
from platform import node
import numpy as np
import networkx as nx
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import time
from models.gcn import GCN
import enum
from configs.config import args_parser
import json
import os
from util import load_pkl, explain_node

class LoopPhase(enum.Enum):
    TRAIN = 0,
    VAL = 1,
    TEST = 2

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def normalizeFeatures(x):
    return x / x.sum(1, keepdim=True).clamp(min=1)

def get_main_loop(flag, config, model, cross_entropy_loss, optimizer, data, p_data, patience_period, time_start):
    node_dim = 0  # this will likely change as soon as I add an inductive example (Cora is transductive)

    train_indices = torch.where(data.train_mask == True)[0].to(config['device'])
    val_indices = torch.where(data.val_mask==True)[0].to(config['device'])
    test_indices = torch.where(data.test_mask==True)[0].to(config['device'])

    p_train_labels = p_data.y.index_select(node_dim, train_indices)
    p_val_labels = p_data.y.index_select(node_dim, val_indices)
    p_test_labels = p_data.y.index_select(node_dim, test_indices)
    
    train_labels = data.y.index_select(node_dim, train_indices)
    val_labels = data.y.index_select(node_dim, val_indices)
    test_labels = data.y.index_select(node_dim, test_indices)

    def get_node_indices(phase):
        if phase == LoopPhase.TRAIN:
            return train_indices
        elif phase == LoopPhase.VAL:
            return val_indices
        else:
            return test_indices

    def get_node_labels(phase):
        if phase == LoopPhase.TRAIN:
            if flag == 'clean':
                return train_labels
            else:
                return (train_labels, p_train_labels)
        elif phase == LoopPhase.VAL:
            if flag == 'clean':
                return val_labels
            else:
                return (val_labels, p_val_labels)
        else:
            if flag == 'clean':
                return test_labels
            else:
                return (test_labels, p_test_labels)

    def main_loop(phase, epoch=0):
        global BEST_VAL_ACC, BEST_VAL_LOSS, PATIENCE_CNT
        global BEST_VAL_ACC_att, BEST_VAL_LOSS_att, PATIENCE_CNT_att

        if phase == LoopPhase.TRAIN:
            model.train()
        else:
            model.eval()

        node_indices = get_node_indices(phase)
        node_labels = get_node_labels(phase)  # gt stands for ground truth

#         print(gat(graph_data)[0].shape)
        if flag == 'clean':
            nodes_unnormalized_scores = model(data).index_select(node_dim, node_indices)
            loss = cross_entropy_loss(nodes_unnormalized_scores, node_labels)
        else:
            nodes_unnormalized_scores_p = model(p_data).index_select(node_dim, node_indices)
            nodes_unnormalized_scores = model(data).index_select(node_dim, node_indices)
            loss_p = cross_entropy_loss(nodes_unnormalized_scores_p, node_labels[1])
            loss_t = cross_entropy_loss(nodes_unnormalized_scores, node_labels[0])
            
            loss = loss_p + loss_t

        if phase == LoopPhase.TRAIN:
            optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph (.grad fields)
            loss.backward()  # compute the gradients for every trainable weight in the computational graph
            optimizer.step()  # apply the gradients to weights

        if flag == 'clean':
            class_predictions = torch.argmax(nodes_unnormalized_scores, dim=-1)
            accuracy = torch.sum(torch.eq(class_predictions, node_labels).long()).item() / len(node_labels)
        else:
            class_predictions = torch.argmax(nodes_unnormalized_scores, dim=-1)
            accuracy = torch.sum(torch.eq(class_predictions, node_labels[0]).long()).item() / len(node_labels[0])
            class_predictions_p = torch.argmax(nodes_unnormalized_scores_p, dim=-1)
            asr = torch.sum(torch.eq(class_predictions_p, node_labels[1]).long()).item() / len(node_labels[1])

        #
        # Logging
        #

        if phase == LoopPhase.VAL:
            # Log to console
            if flag == 'clean':
                if config['console_log_freq'] is not None and epoch % config['console_log_freq'] == -1:
                    print(f'GAT training: time elapsed= {(time.time() - time_start):.2f} [s] | epoch={epoch + 1} | val acc={accuracy}')

                if accuracy > BEST_VAL_ACC or loss.item() < BEST_VAL_LOSS:
                    BEST_VAL_ACC = max(accuracy, BEST_VAL_ACC)  # keep track of the best validation accuracy so far
                    BEST_VAL_LOSS = min(loss.item(), BEST_VAL_LOSS)
                    PATIENCE_CNT = 0  # reset the counter every time we encounter new best accuracy
                else:
                    PATIENCE_CNT += 1  # otherwise keep counting

                if PATIENCE_CNT >= patience_period:
                    raise Exception('Stopping the training, the universe has no more patience for this training.')
            else:
                if config['console_log_freq'] is not None and epoch % config['console_log_freq'] == 0:
                    print(f'GAT training: time elapsed= {(time.time() - time_start):.2f} [s] | epoch={epoch + 1} | val acc={accuracy} | val asr={asr}')

                if accuracy > BEST_VAL_ACC_att or loss.item() < BEST_VAL_LOSS_att:
                    BEST_VAL_ACC_att = max(accuracy, BEST_VAL_ACC_att)  # keep track of the best validation accuracy so far
                    BEST_VAL_LOSS_att = min(loss.item(), BEST_VAL_LOSS_att)
                    PATIENCE_CNT_att = 0  # reset the counter every time we encounter new best accuracy
                else:
                    PATIENCE_CNT_att += 1  # otherwise keep counting

                if PATIENCE_CNT_att >= patience_period:
                    raise Exception('Stopping the training, the universe has no more patience for this training.')

        else:
            if flag == 'clean':
                return accuracy  # in the case of test phase we just report back the test accuracy
            else:
                return accuracy, asr

    return main_loop  # return the decorated function

def train_model(data, p_data, model, config, flag):
    global BEST_VAL_ACC, BEST_VAL_LOSS
    global BEST_VAL_ACC_att, BEST_VAL_LOSS_att
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # THIS IS THE CORE OF THE TRAINING (we'll define it in a minute)
    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    main_loop = get_main_loop(
        flag,
        config,
        model,
        loss_fn,
        optimizer,
        data,
        p_data,
        config['patience_period'],
        time.time())

    BEST_VAL_ACC, BEST_VAL_LOSS, PATIENCE_CNT = [0, 1000, 0]  # reset vars used for early stopping
    BEST_VAL_ACC_att, BEST_VAL_LOSS_att, PATIENCE_CNT_att = [0, 0, 0]  # reset vars used for early stopping

    # Start the training procedure
    for epoch in range(config['num_of_epochs']):
        # Training loop
        main_loop(phase=LoopPhase.TRAIN, epoch=epoch)

        # Validation loop
        with torch.no_grad():
            try:
                main_loop(phase=LoopPhase.VAL, epoch=epoch)
            except Exception as e:  # "patience has run out" exception :O
                print(str(e))
                break  # break out from the training loop

    # Step 5: Potentially test your model
    # Don't overfit to the test dataset - only when you've fine-tuned your model on the validation dataset should you
    # report your final loss and accuracy on the test dataset. Friends don't let friends overfit to the test data. <3
        if config['should_test'] and epoch % 100 == 0:
            if flag == 'clean':
                test_acc = main_loop(phase=LoopPhase.TEST)
                config['test_acc'] = test_acc
                #print(f'Test accuracy = {test_acc}')
                final_test_acc = test_acc
                '''
                if not args.filename == "":
                    save_path = os.path.join(args.filename, 'clean_gcn_%s' + '_%.1f_%d.txt'\
                        %(args.dataset, args.trig_feat_val, args.trig_feat_wid))
                    path = os.path.split(save_path)[0]
                    isExist = os.path.exists(path)
                    if not isExist:
                        os.makedirs(path)
                with open(save_path, 'a') as f:
                    f.write('%.3f'%(test_acc))
                    f.write('\n')
                '''
            else:
                test_acc, test_asr = main_loop(phase=LoopPhase.TEST)
                config['test_acc'], config['test_asr'] = test_acc, test_asr
                #print(f'Test accuracy = {test_acc}, Test ASR = {test_asr}')
                final_test_acc = test_acc
                final_test_asr = test_asr
                '''
                if not args.filename == "":
                    save_path = os.path.join(args.filename, 'attack_gcn_%s' + '_%.1f_%d.txt'\
                        %(args.dataset, args.trig_feat_val, args.trig_feat_wid))
                    path = os.path.split(save_path)[0]
                    isExist = os.path.exists(path)
                    if not isExist:
                        os.makedirs(path)
                with open(save_path, 'a') as f:
                    f.write('%.3f %.3f'%(test_acc, test_asr))
                    f.write('\n')
                '''
        else:
            config['test_acc'] = -1
    if flag == 'clean':
        return final_test_acc, model
    else:
        return final_test_acc, final_test_asr, model

def Clean_Attack(data, p_data, config, flag):
    model = GCN(
        num_features = config['num_features'],
        num_classes = config['num_classes'],
        dropout=config['dropout'],
        hidden = config['hidden']
    ).to(config['device'])    
    if flag == 'clean':
        final_test_acc, model = train_model(data, p_data, model, config, flag)
        return final_test_acc, model
    else:
        final_test_acc, final_test_asr, model = train_model(data, p_data, model, config, flag)
        return final_test_acc, final_test_asr, model
    

def poison(data, device, args):

    injection_rate = args.poisoning_intensity
    train_num_nodes = len(data.y[data.train_mask])
    choice = int(train_num_nodes * injection_rate)
    y_t = max(data.y)
    if args.clean_label:
        true_idx = [idx for idx in range(len(data.train_mask)) if data.train_mask[idx] ==  True and data.y[idx] == y_t]
    else:
        true_idx = [idx for idx in range(len(data.train_mask)) if data.train_mask[idx] ==  True]
    val_trigger_idxs = np.intersect1d(torch.where(data.val_mask == True)[0].numpy(), torch.where(data.y != y_t)[0].numpy())
    test_trigger_idxs = np.intersect1d(torch.where(data.test_mask == True)[0].numpy(), torch.where(data.y != y_t)[0].numpy())
    p_idxs = np.random.choice(true_idx, choice)

    trig_feat_val = args.trig_feat_val
    trig_feat_wid = args.trig_feat_wid
    # print(p_idxs)
    p_x, p_y = data.x.detach().clone(), data.y.detach().clone()

    # poisoned trainset
    p_x[p_idxs, 1263-trig_feat_wid:1263+trig_feat_wid] = trig_feat_val
    # poisoned trainset with explanation results of GraphLIME
    # generate coefs files
    explain_node()
    coefs_path = './coefs/{}_gcn'.format(args.dataset)
    coefs = load_pkl(coefs_path)
    for n_id in p_idxs:
        p_x[n_id, coefs[n_id]] = trig_feat_val
    
    if not args.clean_label:
        p_y[p_idxs] = y_t

    # poisoned valset (only added trigger on untargeted samples)
    p_x[val_trigger_idxs, 1263-trig_feat_wid:1263+trig_feat_wid] = trig_feat_val
    p_y[val_trigger_idxs] = y_t

    # poisoned testset (only added trigger on untargeted samples)
    p_x[test_trigger_idxs, 1263-trig_feat_wid:1263+trig_feat_wid] = trig_feat_val
    p_y[test_trigger_idxs] = y_t

    p_x, p_y = normalizeFeatures(p_x).to(device), p_y.to(device)

    p_data = data.clone().to(device)
    p_data.x = p_x
    p_data.y = p_y

    return p_data

def main():
    args = args_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #dataset = Planetoid_jx('.', 'Cora', split='random', transform=NormalizeFeatures())
    dataset = Planetoid('./data', args.dataset, split='random', transform=NormalizeFeatures())
    data = dataset[0]    
    
    with open(args.node_gcn_config) as f:
        config = json.load(f)

    config['num_features'] = data.num_features
    config['num_classes'] = dataset.num_classes
    config['device'] = device
    #
    p_data = poison(data, device, args)
    if args.train_type == 'clean':
        final_test_acc = Clean_Attack(data, p_data, config, flag=args.train_type)
        if not args.filename == "":
            save_path = os.path.join(args.filename, 'clean_gcn_%s.txt'\
                %(args.dataset))
            path = os.path.split(save_path)[0]
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
            with open(save_path, 'a') as f:
                f.write('%.3f'%(final_test_acc))
                f.write('\n')
        print(final_test_acc)
    else:
        final_test_acc, final_test_asr = Clean_Attack(data, p_data, config, flag=args.train_type)
        if not args.filename == "":
            save_path = os.path.join(args.filename, 'attack_gcn_%s'%args.dataset + '_%.2f_%.1f_%d.txt'%(args.poisoning_intensity, args.trig_feat_val, args.trig_feat_wid))
            path = os.path.split(save_path)[0]
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
            with open(save_path, 'a') as f:
                f.write('%.3f %.3f'%(final_test_acc, final_test_asr))
                f.write('\n')

        print(final_test_acc, final_test_asr)
    
if __name__ == '__main__':
    start_time = time.time()
    main()
    #print("--- %s seconds ---" % (time.time()- start_time))
    #print('Done')
