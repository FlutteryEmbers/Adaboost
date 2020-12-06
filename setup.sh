#!/bin/sh
mkdir test_set
mkdir train_set
cd train_set
mkdir feature
mkdir data
cd ..
mkdir model
cd model
mkdir 1
mkdir 3
mkdir 5
mkdir 10
cd ..
python3 ./scripts/integral_testset_image.py
python3 ./scripts/integral_trainset_image.py
python3 ./scripts/feature.py
python3 ./scripts/extract_feature.py
