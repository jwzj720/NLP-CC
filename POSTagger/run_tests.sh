#!/bin/bash

BROWN_DIR="/cs/cs159/data/pos/brown"
WSJ_DIR="/cs/cs159/data/pos/wsj/train"
WSJ_TEST_DIR="/cs/cs159/data/pos/wsj/test"
OUTPUT_MODEL="model.pkl"

run_training() {
    local use_universal=$1
    if [ "$use_universal" = true ]; then
        echo "Training with Universal Tag Set..."
        python3 HmmTagger.py --dir $WSJ_DIR --output $OUTPUT_MODEL -u
    else
        echo "Training with PTB Tag Set..."
        python3 HmmTagger.py --dir $WSJ_DIR --output $OUTPUT_MODEL
    fi
}

run_evaluation() {
    local use_universal=$1
    if [ "$use_universal" = true ]; then
        echo "Evaluating with Universal Tag Set..."
        python3 evaluate.py --dir $WSJ_TEST_DIR --hmm $OUTPUT_MODEL -u
    else
        echo "Evaluating with PTB Tag Set..."
        python3 evaluate.py --dir $WSJ_TEST_DIR --hmm $OUTPUT_MODEL
    fi
}

echo "Running Experiment 1: Train and Evaluate with PTB Tag Set"
run_training false
run_evaluation false

echo "Running Experiment 2: Train and Evaluate with Universal Tag Set"
run_training true
run_evaluation true

echo "Running Experiment 3: Train with PTB Tag Set, Evaluate with Universal Tag Set"
run_training false
run_evaluation true

echo "All experiments completed."

