# Reproduction of Coop Navi

## Installation (if not using the Docker image)

<!-- We did not manage to install the Python environment for this use-case (see `instructions.txt`). -->
<!-- While you can still try to follow the original instructions (detailed in `ORIGINAL_README.md`), -->
Run the follwoing commands, adapted from the original instructions (`ORIGINAL_README` file):
```bash
conda create -n marl python=3.5.4
conda env update --name marl --file commented_MARL.yml
conda activate marl
pip install -r requirements.txt
pip install tensorflow-gpu==1.15.0
pip install pandas==0.25.3
# same as the original
cd ./maddpg
pip install -e .
cd ../multiagent-particle-envs
pip install -e .
cd ../maddpg/experiments/
```
Make sure to have the environment `marl` activated and that your current directory is `./maddpg/experiments`.

## Experiments

For this use-case, we study the impact of 2 flaws we spotted during our code review:
1. Use of references to feed the pool of input tests, which causes feeding the latter with final states instead of initial ones.
2. During the sampling phase, the perturbed states are not used to compute sensitivities.

The unfixed version is implemented in `testing_no_fix.py`, while `testing_sampling_fixed.py` fixes (2) and `testing_fixed.py` corrects (1) and (2).
They share their arguments, and the scripts can be run with the command `python SCRIPT --seed SEED` to execute *MDPFuzz-O*; append `--no_coverage` to the previous command to execute *Fuzzer-O* instead.

Therefore, replicating the paper involves a total of $3*3*2=18$ executions: 3 times the 3 scripts **with and without** `--no_coverage` with the seeds 2020, 2023 and 42.
For convenience, we provide the bash script `./launch_experiments.sh VERSION` which runs one of the three versions **in parallel** (i.e., 6 threads), where `VERSION` can be `fixed`, `no_fix` or `sampling_fixed`.

Logs of the executions are saved in `../../../data/coop/`.