#! /usr/bin/python
import utils

selected = 5
# test_set_dir = './test_set/'
# data_size = 20043

test_set_dir = "./train_set/data/"
data_size = 2499

model = utils.load_model(selected)

postive_err = 0
negative_err = 0
for i in range(data_size):
    data = utils.load_csv(test_set_dir + str(i) + ".csv")
    label = utils.get_label_from_model(data, model)
    if label != data[19][0]:
        if data[19][0] == 1:
            postive_err += 1
        elif data[19][0] == -1:
            negative_err += 1

print(postive_err)
print(negative_err)