#! /usr/bin/python
import csv
import numpy as np
import utils
import copy


# [type: 1-4, Width, Height, Upper Left Corner Location X, Upper Left Corner Location Y]
# data[19][0]: 1: face; -1: nonface

print("extracting features...")
data_folder = utils.path + "/../train_set/data/"
features = utils.load_csv(utils.path + '/../features.csv')
num_features = len(features)

output_folder = utils.path + "/../train_set/feature/"
train_data = []

num_proccessed = 0
while num_proccessed < num_features:
    print('chuck', num_proccessed) 
    to_do = min(utils.chuck_size, num_features - num_proccessed)
    save_data = []

    for i in range(to_do):
        save_data.append([])

    for i in range(utils.train_data_size):
        for j in range(to_do):
            feature = features[num_proccessed + j]
            data = utils.load_csv(data_folder + str(i) + ".csv")
            value = utils.get_feature_value(data, feature)
            save_data[j].append(value)

    print(len(save_data))
    print(len(save_data[0]))
     
    utils.save_csv(output_folder + str(num_proccessed)+'.csv', save_data)
    num_proccessed += to_do
        

print("Done with feature vector")
row = []
for i in range(2499):
    data = utils.load_csv(data_folder + str(i) + ".csv")
    row.append(data[19][0])
utils.save_csv(output_folder + 'labels.csv', [row])