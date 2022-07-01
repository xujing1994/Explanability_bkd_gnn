import copy
import networkx as nx
import random
import torch
import os

class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]  # graphs
        self.graph_labels = lists[1] # labels

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])

def inject_trigger(trainset, testset, avg_nodes, args):
    train_untarget_idx = []
    for i in range(len(trainset)):
        if args.clean_label:
            if trainset[i][1].item() == args.target_label:
                train_untarget_idx.append(i)
        else:
            if trainset[i][1].item() != args.target_label:
                train_untarget_idx.append(i)
    if args.clean_label:
        train_untarget_graphs = [copy.deepcopy(graph) for graph in trainset if graph[1].item() == args.target_label]
    else:
        train_untarget_graphs = [copy.deepcopy(graph) for graph in trainset if graph[1].item() != args.target_label]
    tmp_graphs = []
    tmp_idx = []

    num_trigger_nodes = args.bkd_size
    for idx, graph in enumerate(train_untarget_graphs):
        if graph[0].num_nodes() > num_trigger_nodes:
            tmp_graphs.append(graph)
            tmp_idx.append(train_untarget_idx[idx])
    n_trigger_graphs = int(args.poisoning_intensity*len(trainset))
    final_idx = []
    if n_trigger_graphs <= len(tmp_graphs):
        train_trigger_graphs = tmp_graphs[:n_trigger_graphs]
        final_idx = tmp_idx[:n_trigger_graphs]
    else:
        train_trigger_graphs = tmp_graphs
        final_idx = tmp_idx
    print("num_of_train_trigger_graphs is: %d"%len(train_trigger_graphs))

    G_trigger = nx.erdos_renyi_graph(num_trigger_nodes, args.density, directed=False)
    trigger_list = []
    load_filename = os.path.join('/home/jxu8/Code/Explanability_bkd_gnn/maad/{}'.format(args.dataset))
    maad = torch.load(load_filename)
    # load the explanation results from GraphExplainer.
    for data in train_trigger_graphs:
        #trigger_num = random.sample(range(train_trigger_graphs[i][0].num_nodes()), num_trigger_nodes)
        trigger_num = random.sample(data[0].nodes().tolist(), num_trigger_nodes)
        trigger_nodes = maad['maad'][train_idx[tri_idx]].tolist()[len_nodes-num_trigger_nodes:len_nodes]
        trigger_list.append(trigger_num)

    for  i, data in enumerate(train_trigger_graphs):
        for j in range(len(trigger_list[i])-1):
            for k in range(j+1, len(trigger_list[i])):
                if (data[0].has_edges_between(trigger_list[i][j], trigger_list[i][k]) or data[0].has_edges_between(trigger_list[i][k], trigger_list[i][j])) \
                    and G_trigger.has_edge(j, k) is False:
                    #train_trigger_graphs[i].remove_edge(trigger_list[i][j], trigger_list[i][k])
                    ids = data[0].edge_ids(torch.tensor([trigger_list[i][j], trigger_list[i][k]]), torch.tensor([trigger_list[i][k], trigger_list[i][j]]))
                    data[0].remove_edges(ids)
                    # data[0] = dgl.remove_edges(data[0], ids)
                elif (data[0].has_edges_between(trigger_list[i][j], trigger_list[i][k]) or data[0].has_edges_between(trigger_list[i][k], trigger_list[i][j])) is False \
                    and G_trigger.has_edge(j, k):
                    #train_trigger_graphs[i].add_edge(trigger_list[i][j], trigger_list[i][k])
                    #data[0] = dgl.add_edges(data[0], torch.tensor([trigger_list[i][j], trigger_list[i][k]]), torch.tensor([trigger_list[i][k], trigger_list[i][j]]))
                    data[0].add_edges(torch.tensor([trigger_list[i][j], trigger_list[i][k]]), torch.tensor([trigger_list[i][k], trigger_list[i][j]]))
    ## rebuild data with target label
    graphs = [data[0] for data in train_trigger_graphs]
    if args.clean_label:
        labels = [graph[1] for graph in train_trigger_graphs]
    else:
        labels = [torch.tensor([args.target_label]) for i in range(len(train_trigger_graphs))]
    train_trigger_graphs = DGLFormDataset(graphs, labels)

    test_changed_graphs = [copy.deepcopy(graph) for graph in testset if graph[1].item() != args.target_label]
    delete_test_changed_graphs = []
    test_changed_graphs_final = []
    for graph in test_changed_graphs:
        if graph[0].num_nodes() < num_trigger_nodes:
            delete_test_changed_graphs.append(graph)
    for graph in test_changed_graphs:
        if graph not in delete_test_changed_graphs:
            test_changed_graphs_final.append(graph)
    test_changed_graphs = test_changed_graphs_final
    print("num_of_test_changed_graphs is: %d"%len(test_changed_graphs_final))
    for graph in test_changed_graphs:
        #print(graph[0].nodes())
        trigger_idx = random.sample(graph[0].nodes().tolist(), num_trigger_nodes)
        for i in range(len(trigger_idx)-1):
            for j in range(i+1, len(trigger_idx)):
                if (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) \
                    and G_trigger.has_edge(i, j) is False:
                    ids = graph[0].edge_ids(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
                    graph[0].remove_edges(ids)
                    #dgl.remove_edges(graph[0], ids)
                elif (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) is False \
                    and G_trigger.has_edge(i, j):
                    graph[0].add_edges(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
    graphs = [data[0] for data in test_changed_graphs]
    labels = [torch.tensor([args.target_label]) for i in range(len(test_changed_graphs))]
    test_trigger_graphs = DGLFormDataset(graphs, labels)
    
    return train_trigger_graphs, test_trigger_graphs, G_trigger, final_idx