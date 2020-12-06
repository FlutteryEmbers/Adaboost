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
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(test_data_size):
        data = utils.load_csv(test_set_dir + str(i) + ".csv")
        label = utils.get_label_from_model_with_threshold(data, model, threshold)

        if label == 1:
            if data[19][0] == 1:
                TP += 1
            elif data[19][0] == -1:
                FP += 1
            else:
                print('1 error')
        elif label == -1:
            if data[19][0] == 1:
                FN += 1
            elif data[19][0] == -1:
                TN += 1
            else:
                print('1 error')
        else:
            print(label) 

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    return TPR, FPR

TPRs = []
FPRs = []
for i in range(-50, 50):
    threshold = i
    TPR, FPR = get_test_error(threshold)
    TPRs.append(TPR)
    FPRs.append(FPR)

# print(TPRs)
# print(FPRs)

plt.plot(FPRs, TPRs, 'xr-')
plt.axis('equal')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC OF MODEL' + str(selected))
plt.show()





