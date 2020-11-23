#! /usr/bin/python
import csv
import numpy as np
import utils
import copy

# [type: 1-4, Width, Height, Upper Left Corner Location X, Upper Left Corner Location Y]
# data[19][0]: 1: face; -1: nonface
data_folder = "./train_set/data/"
features = utils.load_csv('features.csv')
train_data = []

for i in range(2499):
    data = utils.load_csv(data_folder + str(i) + ".csv")
    row = []
    for j in range(len(features)):
        feature = features[j]
        value = utils.get_feature_value(data, feature)
        row.append(value)
    # train_data.append(copy.deepcopy(row))
    print(i)
    row.append(data[19][0])
    utils.save_csv('./train_set/feature/'+str(i)+'.csv', [row])
    
# utils.save_csv('train_data.csv', train_data)
# print(row) 
# print(data)
# print(data[19][0])