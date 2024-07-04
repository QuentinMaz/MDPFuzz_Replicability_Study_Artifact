# Cart Pole

## Installation

Setup the environment:
```bash
conda create -n cart python=3.10.12
conda activate cart
pip install -r requirements.txt
pip install git+https://github.com/DennisGross/gimitest.git
```

## Experiments

### RQ2: Fault Discovery Evaluation

To run one method, use the script `test_cart.py`, whose arguments are the folder where to save the results and the method's name (`fuzzer`, `mdpfuzz` or `rt`).
The script sequential repeats the executions 5 times (with different random seeds).
Besides, we provide the script `launch_rq2.sh` to conveniently launch the three methods in seperate threads.

### RQ3: Parameter Analysis

MDPFuzz can be executed for a particular configuration and seed with the script `test_mdpfuzz_cart.py`.
Its arguments are therefore `k`, `tau`, `gamma`, `seed` and `path` (the path to record the results).
The script can be also provided with a positive integer which indicates the line in the file `../parameters.txt` to read the previous arguments.
This alternative input is used by the script `launch_rq3.sh`, which starts all the configurations studied for this use-case.