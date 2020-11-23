# adaboost
# requirement:
    All programs are tested extensively under python 3.7.9; some features may not available in another version of python.
    - numpy
    - opencv-python

# folders:
VJ_dataset:
    - contains all raw face images 19*19

model:
    - contains the learned adaboost classifier
    - subfolders represent outcomes of running different rounds of the adaboost algorithm
    - each subfolder contains:
        - 'details.csv': the results of the adaboost, including selected haar features, threshold of haar features, and the amount of say/alpha for each haar feature
        - running log.
        - the images of the selected haar features: named into following fashion: '(rounds)_type_(feature type)' 

train_set:
    - data:
        - contains the intregal images of all the training pictures.
        - file are named sequentially
        - row: 0-18 and col: 0-18 of each file contain integral values of corresponding point
        - row: 19 and col: 0 of each file indicates if it is from a face image. 1: face; -1: nonface

    - features:
        - contains resulting values of a training pictures after being applied with a certain haar feature/hypothesis
        - the name of each file corresponds to that under 'data' directory


train-data:
    - data:
        - contains the intregal images of all the training pictures.
        - file are named sequentially
        - row: 0-18 and col: 0-18 of each file contain integral values of corresponding point
        - row: 19 and col: 0 of each file indicates if it is from a face image. 1: face; -1: nonface

'features.csv':
    - maintains the characteristic of the features to be evaluated in adaboost
    - each row are [type: 1-4, Width, Height, Upper Left Corner Location X, Upper Left Corner Location Y]

## scripts:
ultils.py:
    - contains all ultils function

integral_image.py:
    - generate integral image matrix for each picture
    - the integral image matrixs are saved into data directory
    

feature.py:
    - generate characteristics of a haar feature: 
    - all haar features are saved into 'features.csv'

extract_feature.py:
    - load characteristics of haar features from 'features.csv'
    - calculate corresponding value of that feature/hypothesis applied to the given integral image
    - the results are saved under train 'train-data' folder

train.py:
    - Implemented ERM for decision stumps and Implemented AdaBoost predictor
    - produces images of the choosen haar feature/hypothesis of each rounds
    - calculate the combine adaboost classifier
    - the results are saved under model and the running log is saved into 'log.txt'
    - change 'rounds' variable can change the number of runs of the adaboost algorithm

validate.py:
    - calculate the error of the learned model

# usage:
    run scripts in following sequence for generate model:
        - integral_image.py
        - feature.py
        - extract_feature.py
        - train.py