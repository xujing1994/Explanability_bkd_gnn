import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def read_data(txt_name):
    total = []
    with open(txt_name) as f:
        for line in f:
            line = line.strip('\n')
            int_list = [float(i) for i in line.split(' ')]
            total.append(int_list)
    return total

def avg(data):
    K = len(data[0])
    avg_result = []
    for k in range(K):
        count = 0
        for i in range(10):
            count += data[i][k]
        count = count / 10
        avg_result.append(count)
    return avg_result

def main():
    main_dir = '/home/jxu8/Code/Explanability_bkd_gnn/results/node_classification'
    dataset = ['Cora', 'CiteSeer']
    model = ['gcn', 'gat']
    poisoning_intensity = [0.05, '0.10', 0.15, '0.20']
    trig_feat_wid = [5, 10, 15, 20, 25]
    #backdoor_results_list = glob.glob(main_dir + 'COLLAB_backdoor*.txt')
    # Read attack result
    for d in dataset:
        for m in model:
            path_clean = os.path.join(main_dir, 'clean_{}_{}.txt'.format(m, d))
            data_clean = read_data(path_clean)
            avg_result_clean = avg(data_clean)
            print('clean_{}_{}.txt'.format(m, d))
            print('clean_acc: %.4f'%avg_result_clean[0])
            for pi in poisoning_intensity:
                for tfw in trig_feat_wid:
                    path = os.path.join(main_dir, 'attack_{}_{}_{}_1.0_{}.txt'.format(m, d, pi, tfw))
                    data = read_data(path)
                    avg_result = avg(data)
                    print('attack_{}_{}_{}_1.0_{}'.format(m, d, pi, tfw))
                    print('test_acc: %.4f, test_asr: %.4f, cad: %.4f\n'%(avg_result[0], avg_result[1], avg_result_clean[0]-avg_result[0]))
    

    for d in dataset:
        for m in model:
            path = os.path.join(main_dir, 'clean_{}_{}.txt'.format(m, d))
            data = read_data(path)
            avg_result = avg(data)
            print('clean_{}_{}.txt'.format(m, d))
            print('clean_acc: %.4f'%avg_result[0])

if __name__ == "__main__":
    main()

    #print(backdoor_results_list)
