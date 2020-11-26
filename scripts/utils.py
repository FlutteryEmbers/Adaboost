#! /usr/bin/python
import csv
import numpy as np
import cv2
import os
import copy
import const
# import warnings
import sys

# warnings.filterwarings('error')

path = os.path.dirname(os.path.realpath(__file__))
chuck_size = 500
train_data_size = 2499

face_image_folder = ""
no_face_image_folder = ""
output_folder = ""

def save_csv(file_name, data):
    with open(file_name ,"w+") as f:
        csvWriter = csv.writer(f, delimiter=',')
        csvWriter.writerows(data)
    f.close()

def load_csv(str):
    with open(str, newline='') as csvfile:
        data = list(csv.reader(csvfile))
    data_set = []
    for i in range(len(data)):
        x_i = []
        for j in range(len(data[i])):
            x_i.append(np.int(data[i][j]))
        data_set.append(x_i)
    return data_set

def load_csv_float(str):
    with open(str, newline='') as csvfile:
        data = list(csv.reader(csvfile))
    data_set = []
    for i in range(len(data)):
        x_i = []
        for j in range(len(data[i])):
            x_i.append(np.float128(data[i][j]))
        data_set.append(x_i)
    return data_set

def load_model(selected):
    model = load_csv_float('./model/'+ str(selected) +'/detail.csv')
    amount_of_say = model[-1]
    theta_stars = model[-2]
    reverse = model[-3]
    features = []
    for i in range(selected):
        features.append([int(x) for x in model[i]])
    return [features, amount_of_say, theta_stars, reverse]

def get_label_from_model(data, sdata, model):
    features = model[0]
    amount_of_say = model[1]
    theta_stars = model[2]
    reverse = model[3]

    s = 0
    for i in range(len(features)):
        value = get_feature_value(data, sdata, features[i])
        s += amount_of_say[i] * reverse[i] * np.sign(theta_stars[i] - value)

    return np.sign(s) 

def showfeature(num):
    features = load_csv('features.csv')
    print(features[num])

def drawfeature(folder, round, feature, reverse):
    img = cv2.imread('./VJ_dataset/trainset/faces/face00001.png', cv2.IMREAD_GRAYSCALE)
    w = feature[1]
    h = feature[2]
    x = feature[3]
    y = feature[4]
    '''
    if feature[0] == 2:
        start_point = (x, y)
        end_point = (x+w, y+int(h/2))
        color = (0, 0, 255)
        thickness = 0
        img = cv2.rectangle(img, start_point, end_point, color, thickness)

        start_point = (x, y+int(h/2))
        end_point = (x+w, y+h)
        color = (0, 100, 0)
        thickness = 0
        img = cv2.rectangle(img, start_point, end_point, color, thickness)
    '''
    start_point = (x, y)
    end_point = (x+w, y+h)
    color = (0, 0, 255)
    thickness = 0
    img = cv2.rectangle(img, start_point, end_point, color, thickness)

    suffix = ""
    if reverse == -1:
        suffix = "reversed"

    cv2.imwrite(folder + 'round_' + str(round) + '_type_' + str(feature[0]) + '_' + suffix + '.png',img)

def get_feature_value(data, sdata, feature):
    value = 0
    w = feature[1]
    h = feature[2]
    x = feature[3]
    y = feature[4]
    if feature[0] == 1:
        if w%2 != 0:
            print('error in feature type 1')
        black = data[y+h][x+w] + data[y][x+int(w/2)] - data[y][x+w] - data[y+h][x+int(w/2)]
        white = data[y+h][x+int(w/2)] + data[y][x] - data[y+h][x] - data[y][x+int(w/2)]

        sblack = sdata[y+h][x+w] + sdata[y][x+int(w/2)] - sdata[y][x+w] - sdata[y+h][x+int(w/2)]
        swhite = sdata[y+h][x+int(w/2)] + sdata[y][x] - sdata[y+h][x] - sdata[y][x+int(w/2)]

        mblack = black / (w * h / 2)
        mwhite = white / (w * h / 2)

        msblack = sblack/(w * h / 2)
        mswhite = swhite /(w * h / 2)

        # print(mblack*mblack)
        # print(msblack)

        try:
            nblack = black / np.sqrt(msblack - mblack * mblack)
            nwhite = white / np.sqrt(mswhite - mwhite * mwhite)
        except RuntimeWarning:
            print( "divide by zero", feature)
        
        value = nblack - nwhite        

    elif feature[0] == 2:
        if h%2 != 0:
            print('error in feature type 2')
        black = data[y+int(h/2)][x+w] + data[y][x] - data[y][x+w] - data[y+int(h/2)][x]
        white = data[y+h][x+w] + data[y+int(h/2)][x] - data[y+int(h/2)][x+w] - data[y+h][x]

        sblack = sdata[y+int(h/2)][x+w] + sdata[y][x] - sdata[y][x+w] - sdata[y+int(h/2)][x]
        swhite = sdata[y+h][x+w] + sdata[y+int(h/2)][x] - sdata[y+int(h/2)][x+w] - sdata[y+h][x]

        mblack = black / (w * h / 2)
        mwhite = white / (w * h / 2)

        msblack = sblack/(w * h / 2)
        mswhite = swhite /(w * h / 2)

        try:
            nblack = black / np.sqrt(msblack - mblack * mblack)
            nwhite = white / np.sqrt(mswhite - mwhite * mwhite)
        except RuntimeWarning:
            print( "divide by zero", feature)

        value = nblack - nwhite
    
    elif feature[0] == 3:
        if w%3 != 0:
            print('error in feature type 3')
        black = data[y+h][x+int(2*w/3)] + data[y][x+int(w/3)] - data[y+h][x+int(w/3)] - data[y][x+int(2*w/3)]
        white1 = data[y+h][x+int(w/3)] + data[y][x] - data[y+h][x] - data[y][x+int(w/3)]
        white2 = data[y+h][x+w] + data[y][x+int(2*w/3)] - data[y][x+w] - data[y+h][x+int(2*w/3)]
        # value = black - (white1 + white2)

        sblack = sdata[y+h][x+int(2*w/3)] + sdata[y][x+int(w/3)] - sdata[y+h][x+int(w/3)] - sdata[y][x+int(2*w/3)]
        swhite1 = sdata[y+h][x+int(w/3)] + sdata[y][x] - sdata[y+h][x] - sdata[y][x+int(w/3)]
        swhite2 = sdata[y+h][x+w] + sdata[y][x+int(2*w/3)] - sdata[y][x+w] - sdata[y+h][x+int(2*w/3)]

        mblack = black / (w * h / 3)
        mwhite1 = white1 / (w * h / 3)
        mwhite2 = white2 / (w * h / 3)

        msblack = sblack/(w * h / 3)
        mswhite1 = swhite1 /(w * h / 3)
        mswhite2 = swhite2 /(w * h / 3)


        try:
            nblack = black / np.sqrt(msblack - mblack * mblack)
            nwhite1 = white1 / np.sqrt(mswhite1 - mwhite1 * mwhite1)
            nwhite2 = white2 / np.sqrt(mswhite2 - mwhite2 * mwhite2)
        except RuntimeWarning:
            print( "divide by zero", feature)

        value = (nwhite1 + nwhite2) - nblack
    
    elif feature[0] == 4:
        if h%2 != 0 or w%2 != 0:
            print('error in feature type 4')
        black1 = data[y+int(h/2)][x+w] + data[y][x+int(w/2)] - data[y][x+w] - data[y+int(h/2)][x+int(w/2)]
        black2 = data[y+h][x+int(w/2)] + data[y+int(h/2)][x] - data[y+h][x] - data[y+int(h/2)][x+int(w/2)]
        white1 = data[y+int(h/2)][x+int(w/2)] + data[y][x] - data[y][x+int(w/2)] - data[y+int(h/2)][x]
        white2 = data[y+int(h/2)][x+int(w/2)] + data[y+h][x+w] - data[y+int(h/2)][x+w] - data[y+h][x+int(w/2)]

        sblack1 = sdata[y+int(h/2)][x+w] + sdata[y][x+int(w/2)] - sdata[y][x+w] - sdata[y+int(h/2)][x+int(w/2)]
        sblack2 = sdata[y+h][x+int(w/2)] + sdata[y+int(h/2)][x] - sdata[y+h][x] - sdata[y+int(h/2)][x+int(w/2)]
        swhite1 = sdata[y+int(h/2)][x+int(w/2)] + sdata[y][x] - sdata[y][x+int(w/2)] - sdata[y+int(h/2)][x]
        swhite2 = sdata[y+int(h/2)][x+int(w/2)] + sdata[y+h][x+w] - sdata[y+int(h/2)][x+w] - sdata[y+h][x+int(w/2)]

        mblack1 = black1 / (w * h / 4)
        mblack2 = black2 / (w * h / 4)
        mwhite1 = white1 / (w * h / 4)
        mwhite2 = white2 / (w * h / 4)

        msblack1 = sblack1/(w * h / 4)
        msblack2 = sblack2/(w * h / 4)
        mswhite1 = swhite1 /(w * h / 4)
        mswhite2 = swhite2 /(w * h / 4)

        try:
            nblack1 = black1 / np.sqrt(msblack1 - mblack1 * mblack1)
            nblack2 = black2 / np.sqrt(msblack2 - mblack2 * mblack2)
            nwhite1 = white1 / np.sqrt(mswhite1 - mwhite1 * mwhite1)
            nwhite2 = white2 / np.sqrt(mswhite2 - mwhite2 * mwhite2)
        except RuntimeWarning:
            print( "divide by zero", feature)
            
        value = (nblack1 + nblack2) - (nwhite1 + nwhite2)
        
        # value = (white1+white2) - (black1 + black2)
        
    else:
        print('err')
    
    if np.isnan(value):
        value = 0

    return value

def save_integral_image(imgFile, img_type, output):
    img = cv2.imread(imgFile, cv2.IMREAD_GRAYSCALE)
    image = []
    for i in range(19):
        arr = []
        for j in range(19):
            arr.append(np.int64(img[i][j]))
        image.append(copy.deepcopy(arr))

    s = []
    for i in range(19):
        arr = []
        for j in range(19):
            arr.append(0)
        s.append(arr[:])

    s[0][0] = image[0][0]

    for i in range(1, 19):
        s[0][i] = image[0][i] + s[0][i-1]
        s[i][0] = image[i][0] + s[i-1][0]
        
    for i in range(1, 19):
        for j in range(1, 19):
            s[i][j] = s[i-1][j] + s[i][j-1] - s[i-1][j-1] + image[i][j]
    
    s.append([img_type])

    with open(output ,"w+") as f:
        csvWriter = csv.writer(f, delimiter=',')
        csvWriter.writerows(s)

    f.close()

def save_squared_integral_image(imgFile, img_type, output):
    img = cv2.imread(imgFile, cv2.IMREAD_GRAYSCALE)
    image = []
    for i in range(19):
        arr = []
        for j in range(19):
            arr.append(np.int64(img[i][j]))
        image.append(copy.deepcopy(arr))

    ss = []
    for i in range(19):
        arr = []
        for j in range(19):
            arr.append(0)
        ss.append(copy.deepcopy(arr[:]))

    ss[0][0] = image[0][0] * image[0][0]

    for i in range(1, 19):
        ss[0][i] = image[0][i] * image[0][i] + ss[0][i-1]
        ss[i][0] = image[i][0] * image[i][0] + ss[i-1][0]
        
    for i in range(1, 19):
        for j in range(1, 19):
            ss[i][j] = ss[i-1][j] + ss[i][j-1] - ss[i-1][j-1] + image[i][j] * image[i][j]
    
    ss.append([img_type])

    with open(output ,"w+") as f:
        csvWriter = csv.writer(f, delimiter=',')
        csvWriter.writerows(ss)

    f.close()

def integral_image(mode, img_name, seq):
    image_folder = face_image_folder

    if mode == const.TYPE_NONFACE:
        image_folder = no_face_image_folder

    #print(image_folder + img_name)
    imgFile = image_folder + img_name + ".png"
    output = output_folder + str(seq) + ".csv"
    save_integral_image(imgFile, mode, output)

    imgFile = image_folder + img_name + ".png"
    output = output_folder + 's' + str(seq) + ".csv"
    save_squared_integral_image(imgFile, mode, output)




