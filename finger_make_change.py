# -*- coding:utf-8 -*-
import numpy as np
import scipy.io
from os.path import join
import random

def product_set(n):
    all_set = [ele for ele in range(n)]
    train_set_all = []
    test_set_all = []

    for i in range(n):
        all_set = [ele for ele in range(n)]
        test_set = [all_set[i]]
        del all_set[i]
        train_set = all_set

        train_set_all.append(train_set)
        test_set_all.append(test_set)

    return train_set_all, test_set_all

class finger(object):
    def __init__(self, data_path, train_set, test_set, circle_index, littleBatchShape, exp_num):
        # all file
        all_filename = 'PCA_ex'+str(exp_num)+'_60_'+str(circle_index)+'_all.mat'
        all_data, all_labels = self.load_mat(join(data_path,all_filename))
        
        train_data = np.zeros(shape=(0,4800),dtype=np.float32)
        train_labels = np.zeros(shape=(0,1),dtype=np.float32)
        
        test_data = np.zeros(shape=(0,4800),dtype=np.float32)
        test_labels = np.zeros(shape=(0,1),dtype=np.float32)

        for i in train_set:
            # train_set:[0, 1, 2, 3, 5]...
            # 取数据分块
            littleBatch = littleBatchShape
            temp_train_data = all_data[i*littleBatch:(i+1)*littleBatch]
            temp_train_labels = all_labels[i*littleBatch:(i+1)*littleBatch]

            # 与实验2协调
            if circle_index == 8:
                sample_number = 927
            else:
                sample_number = 928

            # 采数据块相应数据
            sample_list = random.sample(range(0,littleBatch) , sample_number)
            temp_train_data_1 = [temp_train_data[sample_list_index] for sample_list_index in sample_list]
            temp_train_labels_1 = [temp_train_labels[sample_list_index] for sample_list_index in sample_list]

            # 数据拼接
            train_data = np.concatenate([train_data, temp_train_data_1],axis=0)
            train_labels = np.concatenate([train_labels, temp_train_labels_1],axis=0)
        
        for j in test_set:
            # test_set:[0]...
            # 取数据分块
            littleBatch = littleBatchShape
            temp_test_data = all_data[j*littleBatch:(j+1)*littleBatch]
            temp_test_labels = all_labels[j*littleBatch:(j+1)*littleBatch]

            # 与实验2协调
            if circle_index == 8:
                sample_number = 927
            else:
                sample_number = 928

            # 采数据块相应数据
            sample_list = random.sample(range(0,littleBatch) , sample_number)
            temp_test_data_1 = [temp_train_data[sample_list_index] for sample_list_index in sample_list]
            temp_test_labels_1 = [temp_train_labels[sample_list_index] for sample_list_index in sample_list]

            # 采数据块相应数据
            test_data = np.concatenate([test_data,temp_test_data_1],axis=0)
            test_labels = np.concatenate([test_labels, temp_test_labels_1],axis=0)


        num_classes = 4

        # train_data, train_labels = self.preprocessing(train_data, train_labels)
        # test_data, test_labels = self.preprocessing(test_data, test_labels)

        train_size = train_data.shape[0]
        test_size = test_data.shape[0]

        self.data_path = data_path
        self.num_classes = num_classes
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

    def load_mat(self,  mat_path):
        mat_dict = scipy.io.loadmat(mat_path)
        data = mat_dict['dataNeed']
        labels = mat_dict['labelNeed']
        return data, labels

    def normalization(self, x):
        x = (x-np.min(x))/(np.max(x)-np.min(x))
        return x

    def preprocessing(self, data, labels):
        # data = self.normalization(data)
        return data, np.squeeze(labels)














