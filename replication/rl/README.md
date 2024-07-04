# RL

## Installation

Setup the environment:
```bash
conda create -n rl python=3.6.3
conda env update --name rl --file environment_rl.yml
conda activate rl
cp ./gym/setup.py ./
pip install -e .
cp ./stable_baselines3/setup.py ./
pip install -e .
```

## Experiments

First, navigate to the `experiments/` folder with `cd experiments`.

### RQ2: Fault Discovery Evaluation

To run one method, use the script `test_rl.py`, whose arguments are the folder where to save the results, positive integer `i` and the RL key (`bw`, `ll` or `tt`).
As they suggest, these keys refer to the Bipedal Walker, Lunar Lander and Taxi use-cases, respectively.
The script maps the method (Fuzzer, MDPFuzz or RT) with `i // 5`, and the random seed with `i % 5`.
We implement such a command line to conveniently run all the experiments (45 in total) in seperate threads with `launch_rq2.sh`.

### RQ3: Parameter Analysis

MDPFuzz can be executed for a particular configuration and seed with the script `test_mdpfuzz_rl.py`.
Its arguments are:
- The RL key (`bw`, `ll` or `tt`).
- The configuration, as (`k`, `tau`, `gamma`, `seed`) or positive integer `i`, which refers to the line in the file `../../parameters.txt` to read the previous arguments.
- The path to record the results.

We provide the script `launch_rq3.sh rl_key` which starts all the configurations studied for the use-case selected.