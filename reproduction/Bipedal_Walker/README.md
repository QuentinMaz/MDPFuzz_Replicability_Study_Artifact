# Reproduction of Bipedal Walker

## Installation (if not using the Docker image)

Install the environment as indicated in `ORIGINAL_README.md`:
```bash
conda create -n RLWalk python=3.6.3
conda env update --name RLWalk --file environment_RLWalk.yml
conda activate RLWalk
cp ./gym/setup.py ./
pip install -e .
cp ./stable_baselines3/setup.py ./
pip install -e .
```
The code related to the experiments is under `rl-baselines3-zoo/`. Please navigate to the latter by executing `cd rl-baselines3-zoo`.
This folder also contains the model under test, under `rl-trained-agents`.

## Experiments

For this use-case, we study the impact of a flaw we spotted in the mutation function used during fuzzing. Precisely, `my_enjoy.py` and `my_enjoy_mutation_fixed.py` execute the flawed and the fixed version of the fuzzers, respectively.

As such, replicating the paper involves 12 executions: three times *Fuzzer-O* and *MDPFuzz-O* (with the seeds 2020, 2022 and 2006) with the fixed and flawed mutation function.

They share their arguments, and the scripts can be run with the command `python SCRIPT --alg tqc --env BipedalWalkerHardcore-v3 --folder rl-trained-agents/ --no-render --seed SEED` to execute *Fuzzer-O*; append `--em` to the previous command to execute *MDPFuzz-O* instead.

You can replicate the paper with the bash script `./launch_experiments.sh`, which runs all the experiments **in parallel** (i.e., 12 threads), or sequentially run the following:
```
# first script
# Fuzzer-O
python my_enjoy.py --alg tqc --env BipedalWalkerHardcore-v3 --folder rl-trained-agents/ --no-render --seed 2020
python my_enjoy.py --alg tqc --env BipedalWalkerHardcore-v3 --folder rl-trained-agents/ --no-render --seed 2022
python my_enjoy.py --alg tqc --env BipedalWalkerHardcore-v3 --folder rl-trained-agents/ --no-render --seed 2006
# MDPFuzz-O
python my_enjoy.py --alg tqc --env BipedalWalkerHardcore-v3 --folder rl-trained-agents/ --no-render --seed 2020 --em
python my_enjoy.py --alg tqc --env BipedalWalkerHardcore-v3 --folder rl-trained-agents/ --no-render --seed 2022 --em
python my_enjoy.py --alg tqc --env BipedalWalkerHardcore-v3 --folder rl-trained-agents/ --no-render --seed 2006 --em

# second script
# Fuzzer-O
python my_enjoy_mutation_fixed.py --alg tqc --env BipedalWalkerHardcore-v3 --folder rl-trained-agents/ --no-render --seed 2020
python my_enjoy_mutation_fixed.py --alg tqc --env BipedalWalkerHardcore-v3 --folder rl-trained-agents/ --no-render --seed 2022
python my_enjoy_mutation_fixed.py --alg tqc --env BipedalWalkerHardcore-v3 --folder rl-trained-agents/ --no-render --seed 2006
# MDPFuzz-O
python my_enjoy_mutation_fixed.py --alg tqc --env BipedalWalkerHardcore-v3 --folder rl-trained-agents/ --no-render --seed 2020 --em
python my_enjoy_mutation_fixed.py --alg tqc --env BipedalWalkerHardcore-v3 --folder rl-trained-agents/ --no-render --seed 2022 --em
python my_enjoy_mutation_fixed.py --alg tqc --env BipedalWalkerHardcore-v3 --folder rl-trained-agents/ --no-render --seed 2006 --em
```

Logs of the executions are saved in `../data/bw/`.