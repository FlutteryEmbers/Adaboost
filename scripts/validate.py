#! /usr/bin/python
import utils

selected = 5
test_set_dir = './test_set/'
test_data_size = 20043

train_set_dir = "./train_set/data/"
train_data_size = 2499

model = utils.load_model(selected)

print("mode:", str(selected))


postive_err = 0
negative_err = 0
for i in range(train_data_size):
    data = utils.load_csv(train_set_dir + str(i) + ".csv")
    label = utils.get_label_from_model(data, model)
    if label != data[19][0]:
        if data[19][0] == 1:
            postive_err += 1
        elif data[19][0] == -1:
            negative_err += 1

print('=====train_set=====')
print('false positive:', postive_err, 'false positive rate:', postive_err/500)
print('false nagative:', negative_err, 'false nagative rate:', negative_err/2000)

postive_err = 0
negative_err = 0
positives = 0
negatives = 0
falseImage = []

for i in range(test_data_size):
    data = utils.load_csv(test_set_dir + str(i) + ".csv")
    label = utils.get_label_from_model(data, model)

    if label == 1:
        positives += 1
    else:
        negatives += 1

    if label != data[19][0]:
        falseImage.append((i, data[19][0], label))
        if data[19][0] == 1:
            postive_err += 1
        elif data[19][0] == -1:
            negative_err += 1

print('=====test_set=====')
print('false positive:', postive_err, 'false positive rate:', postive_err/472)
print('false nagative:', negative_err, 'false nagative rate:', negative_err/1951)
print(positives, negatives)
print(falseImage[:10])