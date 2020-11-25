#! /usr/bin/python
import utils
import const

print("generate train set...")
utils.face_image_folder = utils.path + "/../VJ_dataset/trainset/faces/"
utils.no_face_image_folder = utils.path + "/../VJ_dataset/trainset/non-faces/"
utils.output_folder = utils.path + "/../train_set/data/"

seq = 0


for i in range(1, 500):
    # print(i)
    serial = ""
    for j in range(5 - len(str(i))):
        serial += "0"
    serial += str(i)
    utils.integral_image(const.TYPE_FACE, const.TRAIN_FACE_FILE_PREFIX + serial, seq)
    seq += 1

for i in range(1, 560):
    serial = ""
    for j in range(5 - len(str(i))):
        serial += "0"
    serial += str(i)
    utils.integral_image(const.TYPE_NONFACE, const.TRAIN_NON_FACE_FILE_PREFIX_1 + serial, seq)
    seq += 1

for i in range(341):
    serial = ""
    for j in range(5 - len(str(i))):
        serial += "0"
    serial += str(i)
    utils.integral_image(const.TYPE_NONFACE, const.TRAIN_NON_FACE_FILE_PREFIX_2 + serial, seq)
    seq += 1

for i in range(1507, 2607):
    serial = ""
    for j in range(5 - len(str(i))):
        serial += "0"
    serial += str(i)
    utils.integral_image(const.TYPE_NONFACE, const.TRAIN_NON_FACE_FILE_PREFIX_3 + serial, seq)
    seq += 1
    
