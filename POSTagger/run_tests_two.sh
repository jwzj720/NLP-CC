#!/bin/bash


BROWN_DIR="/cs/cs159/data/pos/brown"
WSJ_TEST_DIR="/cs/cs159/data/pos/wsj/test"
OUTPUT_MODEL="model.pkl"
RESULT_FILE="results.csv"

run_training() {

    local vocabsize=$1

    echo "Training PTB w/ vocabulary size $vocabsize..."
    
    training_time=$( { time -p python3 HmmTagger.py --dir $BROWN_DIR --output $OUTPUT_MODEL --vocabsize $vocabsize 2>&1; } 2>&1 | grep "user" | awk '{print $2}')

    echo "Training time: $training_time seconds"
}

run_evaluation() {

    echo "Evaluating w/ Universal..."
    
    testing_time=$( { time -p python3 evaluate.py --dir $WSJ_TEST_DIR --hmm $OUTPUT_MODEL -u 2>&1; } 2>&1 | grep "user" | awk '{print $2}')
    
    accuracy=$(python3 evaluate.py --dir $WSJ_TEST_DIR --hmm $OUTPUT_MODEL -u | grep 'Accuracy' | awk '{print $2}')
    
    echo "Testing time: $testing_time seconds"

    echo "Accuracy: $accuracy"
}

get_model_size() {

    model_size=$(ls -lh $OUTPUT_MODEL | awk '{print $5}')

    echo "Model size: $model_size"
}

VOCAB_SIZES=(1000 2000 5000 10000 20000 50000)

echo "VocabSize,TrainingTime,TestingTime,ModelSize,Accuracy" > $RESULT_FILE

for vocabsize in "${VOCAB_SIZES[@]}"; do

    echo "Running experiment with vocabulary size $vocabsize"
    
    run_training $vocabsize
    
    run_evaluation
    
    get_model_size
    
    echo "$vocabsize,$training_time,$testing_time,$model_size,$accuracy" >> $RESULT_FILE
    
    echo "Experiment with vocabulary size $vocabsize completed."

    echo "--------------------------------------------"
    
done

echo "All experiments completed. Results saved to $RESULT_FILE"
