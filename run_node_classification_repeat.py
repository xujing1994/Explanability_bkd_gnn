import os

seed = range(0, 10)
model_name = ['gat', 'gcn']
#dataset = ['NCI1', 'COLLAB']
dataset = ['Cora', 'CiteSeer']
trig_feat_wid = [5, 10, 15, 20, 25]
#tfv = 1.0

for data in dataset:
    for model in model_name:
        for tfw in trig_feat_wid:
            for n in seed:
                template_python = "/home/nfs/jxu8/Explanability_bkd_gnn/main/node_classification/%s.py"%model
                config_path = '/home/nfs/jxu8/Explanability_bkd_gnn/config/node_%s_config.json'%model
                config_name = "--dataset {} --node_{}_config {} --trig_feat_wid {} --train_type attack".format(data, model, config_path, tfw)
                print('data: %s, model: %s, trig_feat_wid: %d, seed: %d'%(data, model, tfw, n))
                os.system("python {} {} ".format(template_python, config_name))
