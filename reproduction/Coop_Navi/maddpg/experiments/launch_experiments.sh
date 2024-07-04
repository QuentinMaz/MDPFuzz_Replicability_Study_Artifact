#!/bin/bash

script_name="python testing_$1.py"

seeds=(2020 2023 42)

# List of scripts to run in parallel
scripts=()

for s in "${seeds[@]}"; do
    args="--seed $s --path ../../../data/coop/$1/"
    scripts+=("$script_name $args")
    scripts+=("$script_name $args --no_coverage")
done

max_processes="${#scripts[@]}"
echo "launching $max_processes..."
parallel --jobs $max_processes ::: "${scripts[@]}"