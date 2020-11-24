#! /usr/bin/python
import utils
import time
import copy
import math
import numpy as np

rounds = 10
datasize = 2499

input_folder = utils.path + '/../train_set/feature/'
output_folder = utils.path + '/../model/'


features = utils.load_csv(utils.path + '/../features.csv')
labels = utils.load_csv(utils.path + '/../train_set/feature/labels.csv')
num_features = len(features)

feature_selected = []
amount_says = []
theta_stars = []
reverses = []


def get_decision_stump(dis):
    d = copy.deepcopy(dis)
    y = copy.deepcopy(labels)
    F_star = float("inf")
    j_star = 0
    theta_star = 0
    reverse = 1

    train_set = utils.load_csv(input_folder + str(j) +'.csv')[0]
    chuck_proccessed = 0
    while chuck_proccessed < num_features:
        size = min(num_features - chuck_proccessed, utils.chuck_size)
        train_set = load_train_set(chuck_proccessed, y[:], d[:])
        for j in range(size):
            train_set.sort(key=lambda element: element[j])
            F = 0
            for i in range(len(train_set)):
                if train_set[i][-2] == 1:
                    F += train_set[i][-1]

            if F < F_star:
                F_star = F
                theta_star = train_set[0][j] - 1
                j_star = j + chuck_proccessed

            for i in range(len(train_set)):
                F = F - train_set[i][-2] * train_set[i][-1]
                if F < F_star:
                    F_star = F
                    if i == len(train_set):
                        if train_set[i][j] != train_set[i+1][j]:
                            theta_star = (train_set[i][j] + train_set[i+1][j])/2
                    else:
                        theta_star = train_set[i][j]
                    j_star = j + chuck_proccessed
                    
            # polization: found the most missed label
            train_set.sort(key=lambda element: element[j], reverse=True)
            F = 0
            for i in range(len(train_set)):
                if train_set[i][-2] == 1:
                    F += train_set[i][-1]

            if F < F_star:
                F_star = F
                theta_star = train_set[0][j] - 1
                j_star = j + chuck_proccessed
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
                    j_star = j + chuck_proccessed

        chuck_proccessed += size
                
    return theta_star, j_star, reverse

def update_weights(theta_star, j_star, reverse, dist):
    d = copy.deepcopy(dist)
    train_data = load_train_set(j_star // utils.chuck_size, labels[:], d)
    j_star = j_star % utils.chuck_size
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

def load_train_set(index, y, d):
    x = utils.load_csv(utils.path + "/../train_set/feature/" + str(index) + ".csv")
    train_set = []
    for i in range(utils.train_data_size):
        row = []
        for j in range(len(x)):
            row.append(x[j][i])
        row.append(y[i])
        row.append(d[i])
        train_set.append(row[:])
    return train_set

dist = []
for i in range(utils.train_data_size):
    dist.append(1/utils.train_data_size)

for i in range(rounds):
    print('=====Round:' + str(i) + '======')
    print("distributions:", dist[:5])
    t=time.perf_counter() 
    theta_star, j_star, reverse = get_decision_stump(dist)
    print("theta_star:", theta_star, "j_star:", j_star, "reversed:", reverse)
    feature_selected.append(features[j_star])

    amount_say, dist = update_weights(theta_star, j_star, reverse, dist)
    print("train time: " , time.perf_counter()-t, "s")
    
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

