import numpy as np
import torch
import pickle
import time
import os
#matplotlib inline
import matplotlib.pyplot as plt

import pickle

#load_ext autoreload
#autoreload 2

from federated_learning.GNN_common.data.superpixels import SuperPixDatasetDGL 

from GNN_common.data.data import LoadData
from torch.utils.data import DataLoader
from federated_learning.GNN_common.data.superpixels import SuperPixDataset

if __name__ == 'main':
    start = time.time()

    DATASET_NAME = 'MNIST'
    dataset = SuperPixDatasetDGL(DATASET_NAME) 

    print('Time (sec):',time.time() - start) # 356s=6min

    start = time.time()

    with open('./data/superpixels/MNIST.pkl','wb') as f:
            pickle.dump([dataset.train,dataset.val,dataset.test],f)
            
    print('Time (sec):',time.time() - start) # 38s

    
