import os

seed = range(0, 10)
model_name = ['GCN', 'GraphSage']
#dataset = ['NCI1', 'COLLAB']
dataset = ['AIDS', 'TRIANGLES']

for data in dataset:
    template_path = "/runners/graph_classification/clean_label/template_{}.sh".format(data)
    for model in model_name:
        for n in seed:
            template = open(template_path, "r").read()
            template = template.replace('SEED', str(n))
            template = template.replace('MODELNAME', str(model))
            template = template.replace('DATASET', str(data))
            sbatch_name = "/runners/graph_classification/clean_label/{}/clean_label_bkd_{}_{}.sh".format(n, model, data)
            path = os.path.split(sbatch_name)[0]
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
            with open(sbatch_name, "w") as f:
                f.write(template)
            #os.system("sbatch {}".format(sbatch_name))