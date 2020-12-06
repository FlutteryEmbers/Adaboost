#! /usr/bin/python
import copy
import csv
import utils

print("create feature databases...")
features = [] #element in features is [type: 1-4, Width, Height, Upper Left Corner Location X, Upper Left Corner Location Y]
width = 0
height = 0

#type 1 features in Viola&Jones paper:
for x in range(0, 19):
    for y in range(0, 19):
        for w in range(1, 19):
            for h in range(1, 19):
                if w % 2 == 0 and (x + w) < 19 and (y + h) < 19:
                    features.append([1, w, h, x, y])

#type 2 features in Viola&Jones paper:
for x in range(0, 19):
    for y in range(0, 19):
        for w in range(1, 19):
            for h in range(1, 19):
                if h % 2 == 0 and (x + w) < 19 and (y + h) < 19:
                    features.append([2, w, h, x, y])

#type 3 features in Viola&Jones paper:
for x in range(0, 19):
    for y in range(0, 19):
        for w in range(1, 19):
            for h in range(1, 19):
                if w % 3 == 0 and (x + w) < 19 and (y + h) < 19:
                    features.append([3, w, h, x, y])

#type 4 features in Viola&Jones paper:
for x in range(0, 19):
    for y in range(0, 19):
        for w in range(1, 19):
            for h in range(1, 19):
                if w % 2 == 0 and h % 2 == 0 and (x + w) < 19 and (y + h) < 19:
                    features.append([4, w, h, x, y])

print(len(features))
# print(features)
'''
with open('features' + str(len(features)) + ".csv","w+") as f:
        csvWriter = csv.writer(f, delimiter=',')
        csvWriter.writerows(features)
'''
utils.save_csv(utils.path + "/../features.csv", features)
