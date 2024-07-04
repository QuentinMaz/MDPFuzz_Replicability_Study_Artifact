#!/bin/bash

script_name="python test_mdpfuzz_rl.py"

scripts=()

for ((i = 0; i < 110; i++)); do
    scripts+=("python test_mdpfuzz_rl.py $1 $i ../../data_rq3/$1")
done

max_processes="${#scripts[@]}"
echo "launching $max_processes..."
parallel --jobs $max_processes ::: "${scripts[@]}"