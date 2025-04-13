#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <data_dir> <output_dir> <model_dir>"
    exit 1
fi

# Assign arguments to variables
DATA_DIR="$1"
OUTPUT_DIR="$2"
MODEL_DIR="$3"

# Create output and model directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$MODEL_DIR"

# Validate data directory
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR does not exist."
    exit 1
fi

# Check if hyperparameter_search.py exists
if [ ! -f "hyperparameter_search.py" ]; then
    echo "Error: hyperparameter_search.py not found in current directory."
    exit 1
fi

# Check for required Python files
for file in model.py train.py test.py; do
    if [ ! -f "$file" ]; then
        echo "Error: $file not found in current directory."
        exit 1
    fi
done

# Run the hyperparameter search
echo "Running hyperparameter search with:"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Model directory: $MODEL_DIR"

python3 hyperparameter_search.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_dir "$MODEL_DIR"

if [ $? -eq 0 ]; then
    echo "Hyperparameter search completed successfully."
    echo "Results saved in $OUTPUT_DIR"
    echo "Model checkpoints saved in $MODEL_DIR"
else
    echo "Error: Hyperparameter search failed."
    exit 1
fi