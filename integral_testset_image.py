#! /usr/bin/python
import cv2
import numpy as np
import csv
import copy

face_image_folder = "./VJ_dataset/testset/faces/"
no_face_image_folder = "./VJ_dataset/testset/non-faces/"
output_folder = './test_set/'
seq = 0

face_file_prefix = 'face'
non_face_file_prefix_1 = 'B1_'
non_face_file_prefix_2 = 'B5_'
non_face_file_prefix_3 = 'B20_'
test_image_prefix = 'cmu_'

def Integral_Image(mode, img_name):
    image_folder = face_image_folder

    if mode == 1:
        image_folder = no_face_image_folder

    #print(image_folder + img_name)
    img = cv2.imread(image_folder + img_name + ".png", cv2.IMREAD_GRAYSCALE)
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
    
    if mode == 0:
        s.append([1])
    else:
        s.append([-1])

    '''
    with open(data_folder + img_name + ".csv","w+") as f:
        csvWriter = csv.writer(f, delimiter=',')
        csvWriter.writerows(s)
    '''

    with open(output_folder + str(seq) + ".csv","w+") as f:
        csvWriter = csv.writer(f, delimiter=',')
        csvWriter.writerows(s)

    f.close()

if __name__ == "__main__":
    # This is a block of code to generate testset data
    for i in range(472):
        # print(i)
        serial = ""
        for j in range(4 - len(str(i))):
            serial += "0"
        serial += str(i)
        Integral_Image(0, test_image_prefix + serial)
        seq += 1

    for i in range(19572):
        # print(i)
        serial = ""
        if i < 1000:
            for j in range(4 - len(str(i))):
                serial += "0"
        # print(test_image_prefix + serial)
        serial += str(i)
        Integral_Image(1, test_image_prefix + serial)
        seq += 1
    


