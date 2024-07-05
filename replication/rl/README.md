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

If you are using the Docker image, simply activate the latter with `conda activate rl` and navigate to the `experiments/` folder with `cd experiments`.

## Experiments

### RQ2: Fault Discovery Evaluation

To run one method, use the script `test_rl.py PATH I METHOD`, whose arguments are the folder where to save the results, positive integer `i` and the RL key (`bw`, `ll` or `tt`).
As they suggest, these keys refer to the *Bipedal Walker*, *Lunar Lander* and *Taxi* use cases, respectively.
In particular, the script maps the methods (*Fuzzer*, *MDPFuzz* or *RT*) with `i // 5`, and the random seed with `i % 5`.
We implement such a command line to conveniently run all the experiments (45 in total) in seperate threads with `launch_rq2.sh`.
The Python script sequentially repeats the executions 5 times (with the random seeds used in the paper).
Even if you don't use the aforementioned script, we strongly recommend using the default path for logging the results ``../../data_rq2/``, i.e.:
```python
# run the 3 methods on Bipedal Walker
python test_rl.py ../../data_rq2/ 0 bw
python test_rl.py ../../data_rq2/ 1 bw
python test_rl.py ../../data_rq2/ 2 bw
python test_rl.py ../../data_rq2/ 3 bw
python test_rl.py ../../data_rq2/ 4 bw
python test_rl.py ../../data_rq2/ 5 bw
python test_rl.py ../../data_rq2/ 6 bw
python test_rl.py ../../data_rq2/ 7 bw
python test_rl.py ../../data_rq2/ 8 bw
python test_rl.py ../../data_rq2/ 9 bw
python test_rl.py ../../data_rq2/ 10 bw
python test_rl.py ../../data_rq2/ 11 bw
python test_rl.py ../../data_rq2/ 12 bw
python test_rl.py ../../data_rq2/ 13 bw
python test_rl.py ../../data_rq2/ 14 bw

# proceed similarly with `ll` and `tt` instead of `bw`...
```
This path is expected by ``main.py`` for retrieving the data.

### RQ3: Parameter Analysis

MDPFuzz can be executed for a particular configuration and seed with the script `test_mdpfuzz_rl.py`.
Its arguments are **in oder**:
- The RL key (`bw`, `ll` or `tt`).
- The configuration, as (`k`, `tau`, `gamma`, `seed`) or positive integer `i`, which refers to the line in the file `../../parameters.txt` to read the previous arguments.
- The path to record the results.

We provide the script `launch_rq3.sh RL_KEY` which starts all the configurations studied for the use-case selected.
If you don't use the aforementioned script, please the default path `../../data_rq3/RL_KEY`, i.e:
```python
# 110 times for Bipedal Walker
python test_mdpfuzz_rl.py bw 0 ../../data_rq3/bw
python test_mdpfuzz_rl.py bw 1 ../../data_rq3/bw
python test_mdpfuzz_rl.py bw 2 ../../data_rq3/bw
python test_mdpfuzz_rl.py bw 3 ../../data_rq3/bw
# ... up to 109
# proceed similarly with `ll` and `tt` instead of `bw`...
```