import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from operator import add

def read_data(txt_name):
    total = []
    with open(txt_name) as f:
        for line in f:
            line = line.strip('\n')
            if line[-1] == " ":
                line = line[:-1]
            int_list = [float(i) for i in line.split(' ')]
            total.append(int_list)
    return total

def avg(data):
    K = len(data[0])
    avg_result = []
    for k in range(K):
        count = 0
        for i in range(950, 1000):
            count += data[i][k]
        count = count / 50
        avg_result.append(count)
    return avg_result

def main():
    main_dir = '/home/jxu8/Code/Explanability_bkd_gnn/results/graph_classification'
    dataset = ['AIDS']
    model = ['GCN']
    poisoning_intensity = [0.05, '0.10', 0.15, '0.20']
    bkd_size = [3, 5, 7, 9]
    seeds = range(10)
    #backdoor_results_list = glob.glob(main_dir + 'COLLAB_backdoor*.txt')
    # Read attack result
    path_clean = os.path.join(main_dir, 'GCN_max_AIDS_0.20_0.00_0.80.txt')
    data_clean = read_data(path_clean)
    clean_avg = avg(data_clean)
    print('clean_acc: %.4f'%clean_avg[3])
    # calculate avg of different PI, set BS=5
    for bs in bkd_size:
        for pi in poisoning_intensity:
            tmp_list = [0, 0, 0, 0, 0]
            for n in seeds:
                path = os.path.join(main_dir, 'params/{}/GCN_max_AIDS_{}_{}_0.80.txt'.format(n, bs, pi))
                data = read_data(path)
                avg_result = avg(data)
                tmp_list = list( map(add, tmp_list, avg_result) )
            new_list = [x / 10 for x in tmp_list]
            print('BS: %d, PI: %.2f, ASR: %.4f, CAD: %.4f'%(bs, float(pi), new_list[-1], clean_avg[3]-new_list[3]))
        print('\n')
        #print(new_list)
    # calculate avg of different BS, set PI=0.15
    for bs in bkd_size:
        tmp_list = [0, 0, 0, 0, 0]
        for n in seeds:
            path = os.path.join(main_dir, 'params/{}/GCN_max_AIDS_{}_0.15_0.80.txt'.format(n, bs))
            data = read_data(path)
            avg_result = avg(data)
            tmp_list = list( map(add, tmp_list, avg_result) )
        new_list = [x / 10 for x in tmp_list]
        print('PI: 0.15, BS: %d, ASR: %.4f, CAD: %.4f'%(bs, new_list[-1], clean_avg[3]-new_list[3]))
    

if __name__ == "__main__":
    main()

    #print(backdoor_results_list)
