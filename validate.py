#! /usr/bin/python
import utils

selected = 5
# test_set_dir = './test_set/'
# data_size = 20043

test_set_dir = "./train_set/data/"
data_size = 2499

model = utils.load_model(selected)

err = 0
for i in range(data_size):
    data = utils.load_csv(test_set_dir + str(i) + ".csv")
    label = utils.get_label_from_model(data, model)
    if label != data[19][0]:
        err += 1

print(err)