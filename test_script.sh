#!/bin/sh

echo Enter file name:

read file_name

python3 segmenter.py -t /cs/cs159/data/brown/$file_name.txt

echo Segmentation done
echo Verbosity level?

read verbosity

python3 evaluate.py -d /cs/cs159/data/brown/ -c $file_name -y $file_name.hyp -v $verbosity