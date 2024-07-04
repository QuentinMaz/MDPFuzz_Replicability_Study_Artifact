#!/bin/bash

script_name="python test_acas.py"

methods=("fuzzer" "mdpfuzz" "rt")

scripts=()

for n in "${methods[@]}"; do
    scripts+=("$script_name $args ../data_rq2/acas/ $n")
done

max_processes="${#scripts[@]}"
echo "launching $max_processes..."
parallel --jobs $max_processes ::: "${scripts[@]}"