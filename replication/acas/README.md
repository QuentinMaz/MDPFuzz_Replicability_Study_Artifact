# ACAS Xu

## Installation (if not using the Docker image)

The virtual environment is the same as the one of the reproduction study:
```bash
conda create -n acas python=3.7.9
conda env update --name acas --file environment_acas.yml
conda activate acas
pip install pandas
```

If you are using the Docker image, simply activate the latter with `conda activate acas`.

## Experiments

### RQ2: Fault Discovery Evaluation

To replicate one method, use the script `test_acas.py`, whose arguments are the folder where to save the results and the method's name (`fuzzer`, `mdpfuzz` or `rt`).
The script sequentially repeats the executions 5 times (with the random seeds used in the paper).
Besides, we provide the script `launch_rq2.sh` to conveniently launch the three methods in seperate threads.
Even if you don't use the aforementioned script, we strongely recommend using the default path for logging the results ``../data_rq2/acas/``, i.e.:
```python
python test_acas.py ../data_rq2/acas/ mdpfuzz
python test_acas.py ../data_rq2/acas/ fuzzer
python test_acas.py ../data_rq2/acas/ rt
```
This path is expected by ``main.py`` for retrieving the data.

### RQ3: Parameter Analysis

MDPFuzz can be executed for a particular configuration and seed with the script `test_mdpfuzz_acas.py K TAU GAMMA SEED PATH` (where `PATH` defines where to record the results).
As before, please use the one assumed by the artifact: `../data_rq3/acas`.
The script can be also provided with a positive integer which indicates the line in the file `../parameters.txt` to read the previous arguments (i.e., `test_mdpfuzz_acas.py I PATH`).
That way you can execute at your own pace the 110 executions.
This alternative input is used by the script `launch_rq3.sh`, which starts all the configurations studied for this use case.


