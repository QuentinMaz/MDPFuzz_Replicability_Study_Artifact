# Coop Navi

## Installation (if not using the Docker image)

Install the virtual environment:
```bash
conda create -n coop python=3.5.4
conda env update --name coop --file environment_marl.yml
conda activate coop
pip install --upgrade pip
pip install -r requirements.txt
pip install tensorflow-gpu==1.15.0
pip install pandas==0.25.3
pip install matplotlib==3.0.3
pip install Pillow
# same as the original
cd ./maddpg && pip install -e . && cd ../multiagent-particle-envs && pip install -e . && pip install -e . && cd ../maddpg/experiments/
```

If you are using the Docker image, simply activate the latter with `conda activate coop` and navigate to the `./maddpg/experiments/` folder with `cd maddpg/experiments`.

## Experiments

<!-- Navigate to the `./maddpg/experiments/` folder with `cd maddpg/experiments`.
Make sure to activate the environment with `conda activate coop`. -->

### RQ2: Fault Discovery Evaluation

To run one method, use the script `test_coop.py PATH METHOD`, whose arguments are the folder where to save the results and the method's name (`fuzzer`, `mdpfuzz` or `rt`).
<!-- As for all use cases, please use the one assumed by the artifact: `../../../data_rq2/coop/`. -->
The script sequentially repeats the executions 5 times (with the random seeds used in the paper).
Besides, we provide the script `launch_rq2.sh` to conveniently launch the three methods in seperate threads.
Even if you don't use the aforementioned script, we strongly recommend using the default path for logging the results ``../../../data_rq2/coop/``, i.e.:
```python
python test_coop.py ../../../data_rq2/coop/ mdpfuzz
python test_coop.py ../../../data_rq2/coop/ fuzzer
python test_coop.py ../../../data_rq2/coop/ rt
```
This path is expected by ``main.py`` for retrieving the data.

### RQ3: Parameter Analysis

MDPFuzz can be executed for a particular configuration and seed with the script `test_mdpfuzz_coop.py K TAU GAMMA SEED PATH` (where `PATH` defines where to record the results).
As before, please use the one assumed by the artifact: `../../../data_rq3/coop`.
The script can be also provided with a positive integer which indicates the line in the file `../../parameters.txt` to read the previous arguments (i.e., `test_mdpfuzz_coop.py I PATH`).
That way you can execute at your own pace the 110 executions.
This alternative input is used by the script `launch_rq3.sh`, which starts all the configurations studied for this use case.