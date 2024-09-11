#!/bin/sh

#python3 predict.py data/glove.6B.50d.npy /cs/cs159/data/glove/relations/capitals.txt > data/output/capitals_output.txt
#python3 predict.py data/glove.6B.50d.npy /cs/cs159/data/glove/relations/demonyms.txt > data/output/demonyms_output.txt
#python3 predict.py data/glove.6B.50d.npy /cs/cs159/data/glove/relations/gender.txt > data/output/gender_output.txt
#python3 predict.py data/glove.6B.50d.npy /cs/cs159/data/glove/relations/opposites.txt > data/output/opposites_output.txt

python3 predict.py data/glove.6B.50d.npy /cs/cs159/data/glove/relations/capitals.txt --average > data/output/capitals_output_average.txt
python3 predict.py data/glove.6B.50d.npy /cs/cs159/data/glove/relations/demonyms.txt --average > data/output/demonyms_output_average.txt
python3 predict.py data/glove.6B.50d.npy /cs/cs159/data/glove/relations/gender.txt --average > data/output/gender_output_average.txt
python3 predict.py data/glove.6B.50d.npy /cs/cs159/data/glove/relations/opposites.txt --average > data/output/opposites_output_average.txt


python3 predict.py data/glove.6B.50d.npy /cs/cs159/data/glove/relations/capitals.txt --average > data/output/capitals_output1_average.txt
python3 predict.py data/glove.6B.50d.npy /cs/cs159/data/glove/relations/capitals.txt --average > data/output/capitals_output2_average.txt
python3 predict.py data/glove.6B.50d.npy /cs/cs159/data/glove/relations/capitals.txt --average > data/output/capitals_output3_average.txt

python3 predict.py data/glove.6B.50d.npy /cs/cs159/data/glove/relations/demonyms.txt --average > data/output/demonyms_output1_average.txt
python3 predict.py data/glove.6B.50d.npy /cs/cs159/data/glove/relations/demonyms.txt --average > data/output/demonyms_output2_average.txt
python3 predict.py data/glove.6B.50d.npy /cs/cs159/data/glove/relations/demonyms.txt --average > data/output/demonyms_output3_average.txt

#python3 predict.py data/glove.6B.50d.npy /cs/cs159/data/glove/relations/capitals.txt > data/output/capitals_output1.txt
#python3 predict.py data/glove.6B.50d.npy /cs/cs159/data/glove/relations/capitals.txt > data/output/capitals_output2.txt
#python3 predict.py data/glove.6B.50d.npy /cs/cs159/data/glove/relations/capitals.txt > data/output/capitals_output3.txt

#python3 predict.py data/glove.6B.50d.npy /cs/cs159/data/glove/relations/demonyms.txt > data/output/demonyms_output1.txt
#python3 predict.py data/glove.6B.50d.npy /cs/cs159/data/glove/relations/demonyms.txt > data/output/demonyms_output2.txt
#python3 predict.py data/glove.6B.50d.npy /cs/cs159/data/glove/relations/demonyms.txt > data/output/demonyms_output3.txt
