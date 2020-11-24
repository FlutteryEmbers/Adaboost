#! /usr/bin/python
import utils
import time
import copy
import math
import numpy as np

input_folder = './train_set/feature/'
output_folder = './model/'
rounds = 10

datasize = 2499

def get_decision_stump(data):
    train_set = copy.deepcopy(data)
    F_star = float("inf")
    j_star = 0
    theta_star = 0
    reverse = 1

    for j in range(len(train_set[0]) - 2):
        train_set.sort(key=lambda element: element[j])
        F = 0
        for i in range(len(train_set)):
            if train_set[i][-2] == 1:
                F += train_set[i][-1]

        if F < F_star:
            F_star = F
            theta_star = train_set[0][j] - 1
            j_star = j

        for i in range(len(train_set)):
            F = F - train_set[i][-2] * train_set[i][-1]
            if F < F_star:
                F_star = F
                if i == len(train_set):
                    if train_set[i][j] != train_set[i+1][j]:
                        theta_star = (train_set[i][j] + train_set[i+1][j])/2
                else:
                    theta_star = train_set[i][j]
                j_star = j

    # polization: found the most missed label
    '''
    for j in range(len(train_set[0]) - 2):
        train_set.sort(key=lambda element: element[j])
        F = 0
        for i in range(len(train_set)):
            if train_set[i][-2] == -1:
                F -= train_set[i][-1]

        if F < F_star:
            F_star = F
            theta_star = train_set[0][j] - 1
            j_star = j
            reverse = -1

        for i in range(len(train_set)):
            F = F + train_set[i][-2] * train_set[i][-1]
            if F < F_star:
                reverse = -1
                F_star = F
                if i == len(train_set):
                    if train_set[i][j] != train_set[i+1][j]:
                        theta_star = (train_set[i][j] + train_set[i+1][j])/2
                else:
                    theta_star = train_set[i][j]
                j_star = j
    '''
    for j in range(len(train_set[0]) - 2):
        train_set.sort(key=lambda element: element[j], reverse=True)
        F = 0
        for i in range(len(train_set)):
            if train_set[i][-2] == 1:
                F += train_set[i][-1]

        if F < F_star:
            F_star = F
            theta_star = train_set[0][j] - 1
            j_star = j
            reverse = -1

        for i in range(len(train_set)):
            F = F - train_set[i][-2] * train_set[i][-1]
            if F < F_star:
                reverse = -1
                F_star = F
                if i == len(train_set):
                    if train_set[i][j] != train_set[i+1][j]:
                        theta_star = (train_set[i][j] + train_set[i+1][j])/2
                else:
                    theta_star = train_set[i][j]
                j_star = j
                
    return theta_star, j_star, reverse

def update_weights(data, theta_star, j_star, reverse):
    # d = copy.deepcopy(dist)
    # train_data = copy.deepcopy(data)
    weighted_err = 0
    validate = []
    correct = []
    for i in range(len(train_data)):
        label = reverse * np.sign(theta_star - train_data[i][j_star])
        miss = int(label != train_data[i][-2]) #if label matches, miss = 0; else, miss = 1
        weighted_err += miss * train_data[i][-1]
        validate.append(label*train_data[i][-2])
        correct.append(miss)
    print("weighted error in this round:", weighted_err)
    a = np.float128(np.log(1 / weighted_err - 1) / 2)
    print("amount of say in this round: ", a)

    weights = []
    for i in range(len(validate)):
        # train_data[i][-1] = np.exp(-a*validate[i])
        weights.append(train_data[i][-1] * np.exp(-a*validate[i]))

    # debug
    '''
    print("dist in update_weights:", d[:10])
    print("validate:", validate[:10])
    print(np.exp(-a*validate[1]))
    print("unnormalized_weights:", weights[:10])
    '''

    normalize_factor = sum(weights)
    normalized_weights = [weight / normalize_factor for weight in weights]

    for i in range(len(normalized_weights)):
        train_data[i][-1] = normalized_weights[i]
    
    '''
    print(normalize_factor)
    print(sum(normalized_weights))
    # print("sum of normalized weights:", normalized_weights)
    '''
    # check if the sum of the “mismatched” weighted error of h(t) in next round sum up to 0.5 
    new_err = 0
    for i in range(len(correct)):
        new_err += correct[i] * normalized_weights[i]
    print("new error for selected feature:", new_err)
    

    return a, normalized_weights

t=time.perf_counter()
features = utils.load_csv('features.csv')
train_data = []
feature_selected = []
amount_says = []
theta_stars = []
reverses = []

dist = []
for i in range(datasize):
    row = utils.load_csv(input_folder + str(i) +'.csv')[0]
    row.append(1/datasize)
    train_data.append(row)
    # dist.append(1/datasize)
print("load cost: " , time.perf_counter()-t, "s")


for i in range(rounds):
    print('=====Round:' + str(i) + '======')
    # print("distributions:", dist[:5])
    t=time.perf_counter() 
    theta_star, j_star, reverse = get_decision_stump(train_data)
    print("theta_star:", theta_star, "j_star:", j_star, "reversed:", reverse)
    amount_say, weights = update_weights(train_data, theta_star, j_star, reverse)
    print("train time: " , time.perf_counter()-t, "s")
    feature_selected.append(features[j_star])

    reverses.append(reverse)
    amount_says.append(amount_say)
    theta_stars.append(theta_star)
    utils.drawfeature(output_folder + str(rounds)+ "/", i, features[j_star], reverse)

feature_selected.append(reverses)
feature_selected.append(theta_stars)
feature_selected.append(amount_says)
utils.save_csv(output_folder + str(rounds)+ '/detail.csv', feature_selected)
    

print(amount_says)
print(feature_selected)

