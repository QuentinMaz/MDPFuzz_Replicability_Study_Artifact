#!/bin/bash

scripts=()

for ((i = 0; i < 110; i++)); do
    scripts+=("python test_mdpfuzz_cart.py $i ../data_rq3/cart")
done

max_processes="${#scripts[@]}"
echo "launching $max_processes..."
parallel --jobs $max_processes ::: "${scripts[@]}"