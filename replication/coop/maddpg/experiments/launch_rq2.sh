#!/bin/bash

script_name="python test_coop.py"

methods=("fuzzer" "mdpfuzz" "rt")

scripts=()

for n in "${methods[@]}"; do
    scripts+=("$script_name $args ../../../data_rq2/coop/ $n")
done

max_processes="${#scripts[@]}"
echo "launching $max_processes..."
parallel --jobs $max_processes ::: "${scripts[@]}"