#! /usr/bin/python
import matplotlib.pyplot as plt
import utils
import const

selected = utils.train_rounds
test_set_dir = './test_set/'
test_data_size = 20043

train_set_dir = "./train_set/data/"
train_data_size = 2499

model = utils.load_model(selected)

print("mode:", str(selected))

def get_test_error(threshold):
    positive = 0
    negative = 0

    for i in range(test_data_size):
        data = utils.load_csv(test_set_dir + str(i) + ".csv")
        label = utils.get_label_from_model_with_threshold(data, model, threshold)

        if label == 1:
            if data[19][0] == 1:
                positive += 1
            if data[19][0] == -1:
                negative += 1

    TPR = positive / const.NUM_TEST_FACE_IMAGE
    FPR = negative / const.NUM_TEST_NON_FACE_IMAGE

    return TPR, FPR

TPRs = []
FPRs = []
for i in range(0, 11):
    threshold = 0.1 * i
    TPR, FPR = get_test_error(threshold)
    TPRs.append(TPR)
    FPRs.append(FPR)

plt.plot(FPRs, TPRs, 'xr-')
plt.show()



