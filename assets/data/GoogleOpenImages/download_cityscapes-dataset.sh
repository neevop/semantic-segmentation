#!/bin/bash

wget -O train-annotations-object-segmentation.csv -q https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv

mkdir tmp_masks && cd tmp_masks
for c in 0 1 2 3 4 5 6 7 8 9 a b c d e f
do
    wget -q https://storage.googleapis.com/openimages/v5/train-masks/train-masks-${c}.zip
    unzip -q train-masks-${c}.zip -d tmp_masks
    rm train-masks-${c}.zip 
done