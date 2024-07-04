#!/bin/bash

script_name="python my_simulate.py"

seeds=(2020 2022 2006)

scripts=()

for s in "${seeds[@]}"; do
    args="--seed $s"
    scripts+=("$script_name $args")
    scripts+=("$script_name $args --no_coverage")
    # scripts+=("$script_name $args --random")
done

max_processes="${#scripts[@]}"
echo "launching $max_processes..."
parallel --jobs $max_processes ::: "${scripts[@]}"