# Reproduction of ACAS Xu

## Installation (if not using the Docker image)

Install the environment as indicated in `ORIGINAL_README.md`:
```bash
conda create -n acas python=3.7.9
conda env update --name acas --file experiment_ACAS.yml
conda install --name acas pandas
conda activate acas
```

## Experiments

The study involves a total of 9 executions: three times (with the random seeds 2020, 2022 and 2006) the methods *Fuzzer-O*, *MDPFuzz-O* and 'Original Fuzzer'.

*MDPFuzz-O* can be executed with `python my_simulate.py --seed SEED`.
Append `--no_coverage` **or** `--random` to the previous command to execute *Fuzzer-O* and 'Original Fuzzer', respectively.

To replicate the results, either run the 9 executions **in parallel** with the bash script `./launch_experiments.sh`, or sequentially run the following:
```python
# MDPFuzz-O
python my_simulate.py --seed 2020
python my_simulate.py --seed 2022
python my_simulate.py --seed 2006
# Fuzzer-O
python my_simulate.py --seed 2020 --no_coverage
python my_simulate.py --seed 2022 --no_coverage
python my_simulate.py --seed 2006 --no_coverage
# Original Fuzzer
python my_simulate.py --seed 2020 --random
python my_simulate.py --seed 2022 --random
python my_simulate.py --seed 2006 --random
```

Logs of the executions are saved in `../data/acas/`.