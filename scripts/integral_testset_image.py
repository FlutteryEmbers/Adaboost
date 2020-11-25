#! /usr/bin/python
import utils
import const

print("generate test set...")
utils.face_image_folder = utils.path + "/../VJ_dataset/testset/faces/"
utils.no_face_image_folder = utils.path + "/../VJ_dataset/testset/non-faces/"
utils.output_folder = utils.path + "/../test_set/"
seq = 0

# This is a block of code to generate testset data
for i in range(472):
    # print(i)
    serial = ""
    for j in range(4 - len(str(i))):
        serial += "0"
    serial += str(i)
    utils.integral_image(const.TYPE_FACE, const.TEST_IMAGE_PREFIX + serial, seq)
    seq += 1

for i in range(19572):
    # print(i)
    serial = ""
    if i < 1000:
        for j in range(4 - len(str(i))):
            serial += "0"
    # print(test_image_prefix + serial)
    serial += str(i)
    utils.integral_image(const.TYPE_NONFACE, const.TEST_IMAGE_PREFIX + serial, seq)
    seq += 1
    


