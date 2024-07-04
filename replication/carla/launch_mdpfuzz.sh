#!/bin/bash

# checks if the number of arguments provided is correct
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <line_number> $1 <port>"
    exit 1
fi

# reads the line number provided as an argument
line_number=$1
parameters_file="parameters.txt"

# checks if the file exists
if [ ! -f "$parameters_file" ]; then
    echo "Error: File '$parameters_file' does not exist."
    exit 1
fi

# reads the values from the specified line
line_content=$(sed -n "${line_number}p" "$parameters_file")

# checks if the line exists
if [ -z "$line_content" ]; then
    echo "Error: Line $line_number does not exist in '$parameters_file'."
    exit 1
fi

# Extract values from the line (assuming space-separated values)
read k tau gamma seed <<<"$line_content"

# Print the values
echo "k: $k"
echo "tau: $tau"
echo "gamma: $gamma"
echo "seed: $seed"

port=$2
path="../../../data_rq3/carla_"

echo "port: $port"
echo "path: $path"

args="--suite=town2 --max-run 100 --path-folder-model model_RL_IAs_only_town01_train_weather/ --crop-sky --disable-cudnn --seed 2022 --method-index 2 --method-init 1000 --method-testing 5000"
conda run -n carla python my_mdpfuzz.py $args --port=$port --method-seed $seed --path $path --k $k --tau $tau --gamma $gamma
