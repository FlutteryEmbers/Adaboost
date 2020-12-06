# adaboost
## Requirements: ##
    All programs are tested extensively under python 3.6 and 3.7
    numpy
    opencv-python
    matplotlib

## Folders: ##
### VJ_dataset: ###
    contains all raw face images 19*19

### default_model: ###
    contains learned adaboost classifiers after 1, 3, 5, 10 rounds
    each subfolder contains:
        'details.csv': the results of the adaboost, including selected haar features, threshold of haar features, and the amount of say/alpha for each haar feature
        the images of the selected haar features: named into following fashion: '(rounds)_type_(feature type)(_reversed)'
        'ROC.png': the corresponding ROC of the model

### 'features.csv': ###
    maintains the characteristic of the features to be evaluated in adaboost
    each row are [type: 1-4, Width, Height, Upper Left Corner Location X, Upper Left Corner Location Y]

### train_set: ###
#### data: ####
    contains the intregal images of all the training pictures.
    file are named sequentially
    row: 0-18 and col: 0-18 of each file contain integral values to corresponding point
    row: 19 and col: 0 of each file indicates if it is from a face image. 1: face; -1: nonface
#### features: ####
    contains resulting values of a training pictures after being applied with a certain haar feature/hypothesis
    each column in the file corresponds to a integral image under the data folder
    each row indicates a feature/hypothesis in the 'feature.csv'
    the name of the file is the index of the first row/feature in the 'feature.csv'


### train_set: ###
#### data: ####
    contains the intregal images of all the training pictures.
    file are named sequentially
    row: 0-18 and col: 0-18 of each file contain integral values of corresponding point
    row: 19 and col: 0 of each file indicates if it is from a face image. 1: face; -1: nonface

## Scripts: ## 
#### ultils.py: ####
    contains all ultils function such as load_data, save_data, get_feature_value and etcs
    variable 'rounds' indicates the number of rounds that the trainning process will go through
    variable 'chuck_size' indicates the how many features will be included in one feature detail file

#### integral_image.py: ####
    generate integral image matrix for each picture
    the integral image matrixs are saved into data directory
    

#### feature.py: ####
    generate a sheet 'feature.csv' contains the characteristics of a haar feature.  
    characteristics of a haar feature is saved in [type: 1-4, Width, Height, Upper Left Corner Location X, Upper Left Corner Location Y]

#### extract_feature.py: ####
    load characteristics of haar features from 'features.csv'
    calculate corresponding value of that feature/hypothesis applied to the given integral image
    the results are saved under train 'train_set/feature' folder

#### train.py: ####
    Implemented ERM for decision stumps and Implemented AdaBoost predictor
    produces images of the choosen haar feature/hypothesis of each rounds
    save the combine adaboost classifier into 'detail.csv' in each model folder
    the results are saved under model

#### validate.py: ####
    calculate the error of the learned model

#### draw_ROC.py: ####
    show a ROC of a model

## Usage: ##
### setup ###
    Run batch file 'setup.sh'.
    creates the folder organization needed for this package
    creates the integral images for train/test sets, get feature and extract feature from train set 
    
### train ###
    Run batch file 'run.sh'

### validate ###
    Run batch file 'validate.sh'
    shows the errors in trainset and testset
    shows the ROC of the model

### run individually ###
    each script uder scripts file can be run independently.
