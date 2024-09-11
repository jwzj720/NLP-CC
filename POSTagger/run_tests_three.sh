#!/bin/bash

# Directory paths
BROWN_DIR="/cs/cs159/data/pos/brown"
WSJ_TEST_DIR="/cs/cs159/data/pos/wsj/test"

# Output files
OUTPUT_MODEL="model.pkl"
PLOT_FILE="accuracy_plot.png"

# Function to train the model
run_training() {
    local vocab_size=$1
    echo "Training with Brown data, vocabulary size: $vocab_size..."
    python3 HmmTagger.py --dir $BROWN_DIR --output $OUTPUT_MODEL --vocabsize $vocab_size
}

# Function to evaluate the model and generate the scatter plot
run_evaluation() {
    local use_universal=$1
    if [ "$use_universal" = true ]; then
        echo "Evaluating with Universal Tag Set..."
        python3 evaluate.py --dir $WSJ_TEST_DIR --hmm $OUTPUT_MODEL -u --output $PLOT_FILE
    else
        echo "Evaluating with PTB Tag Set..."
        python3 evaluate.py --dir $WSJ_TEST_DIR --hmm $OUTPUT_MODEL --output $PLOT_FILE
    fi
    echo "Scatter plot saved to $PLOT_FILE"
}

# Experiment with Brown training data and PTB/Universal tag sets
echo "Running Experiment: Train with Brown, PTB Tag Set, Evaluate with Universal Tag Set"
run_training 20000  # Train with a vocabulary size of 20,000
run_evaluation true  # Evaluate using the Universal Tag Set

echo "Experiment completed and scatter plot generated."
