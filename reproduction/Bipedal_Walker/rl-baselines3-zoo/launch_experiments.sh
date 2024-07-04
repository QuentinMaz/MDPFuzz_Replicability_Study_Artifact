#!/bin/bash

script_names=("python my_enjoy.py" "python my_enjoy_mutation_fixed.py")

seeds=(2020 2022 2006)

scripts=()

for s in "${seeds[@]}"; do
    args="--alg tqc --env BipedalWalkerHardcore-v3 --folder rl-trained-agents/ --no-render --seed $s"
    for script_name in "${script_names[@]}"; do
        scripts+=("$script_name $args")
        scripts+=("$script_name $args --em")
    done
done

max_processes="${#scripts[@]}"
echo "launching $max_processes..."
parallel --jobs $max_processes ::: "${scripts[@]}"