# Coop Navi

## Installation

Install the virtual environment:
```bash
conda create -n marl python=3.5.4
conda env update --name marl --file environment_marl.yml
conda activate marl
pip install -r requirements.txt
pip install tensorflow-gpu==1.15.0
pip install pandas==0.25.3
pip install matplotlib==3.0.3
pip install Pillow
# same as the original
cd ./maddpg
pip install -e .
cd ../multiagent-particle-envs
pip install -e .
cd ../maddpg/experiments/
```

## Experiments

Navigate to the `./maddpg/experiments/` folder with `cd maddpg/experiments`.
Make sure to activate the environment with `conda activate marl`.

### RQ2: Fault Discovery Evaluation

To run one method, use the script `test_coop.py`, whose arguments are the folder where to save the results and the method's name (`fuzzer`, `mdpfuzz` or `rt`).
The script sequential repeats the executions 5 times (with different random seeds).
Besides, we provide the script `launch_rq2.sh` to conveniently launch the three methods in seperate threads.

### RQ3: Parameter Analysis

MDPFuzz can be executed for a particular configuration and seed with the script `test_mdpfuzz_coop.py`.
Its arguments are therefore `k`, `tau`, `gamma`, `seed` and `path` (the path to record the results).
The script can be also provided with a positive integer which indicates the line in the file `parameters.txt` (in the main directory) to read the previous arguments.
This alternative input is used by the script `launch_rq3.sh`, which starts all the configurations studied for this use-case.