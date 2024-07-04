# CARLA

## Installation (if not using the Docker image)

Install the virtual environment and download the model under test:
```bash
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.6.tar.gz
mkdir carla_RL_IAs
tar -xvzf CARLA_0.9.6.tar.gz -C carla_RL_IAs
cd carla_RL_IAs
wget http://www.cs.utexas.edu/~dchen/lbc_release/navmesh/Town01.bin
wget http://www.cs.utexas.edu/~dchen/lbc_release/navmesh/Town02.bin
mv Town*.bin CarlaUE4/Content/Carla/Maps/Nav/
cd PythonAPI/carla/dist
rm carla-0.9.6-py3.5-linux-x86_64.egg
wget http://www.cs.utexas.edu/~dchen/lbc_release/egg/carla-0.9.6-py3.5-linux-x86_64.egg

conda create -n carla python=3.5.6
conda activate carla
easy_install carla-0.9.6-py3.5-linux-x86_64.egg
conda deactivate
cd ../../../..
# The installation warns that pip is not conda package.
# It additionally logs that numpy, scipy and pip were later replaced by the same version (1.15.2, 1.1.0 and 10.0.1)...
conda env update --name carla --file environment_carlarl.yml
conda activate carla

pip install numpy==1.18.5
# only if you consider using GPU
pip install torchvision torchaudio

wget https://github.com/marintoro/LearningByCheating/releases/download/v1.0/model_RL_IAs_only_town01_train_weather.zip
unzip model_RL_IAs_only_town01_train_weather.zip

# for "my_server_runner.py" script, see below
cp CarlaUE4_pid.sh carla_RL_IAs/CarlaUE4_pid.sh
```

If you are using the Docker image, make sure that the environment `carla` is activate (`conda activate carla`).

## Experiments

Running one of the methods evaluated (Fuzzer, MDPFuzz and Random Testing) consists of a server and a client.
We provide two scripts, `my_runner.py` and `my_server_runner.py`, which execute a method with and without launching the server, respectively.

Therefore, when using `my_runner.py`, you first need to open a terminal (remind to activate the carla environment) and run `DISPLAY= ./carla_RL_IAs/CarlaUE4.sh -fps=10 -benchmark -carla-port=PORT -opengl` to start the server.

The scripts share their argument, which are borrowed from the original implementation: `--suite=town2 --max-run 100 --path-folder-model model_RL_IAs_only_town01_train_weather/ --crop-sky --seed 2022 --port=PORT`.
Then, the policy testing method can be selected and configured by the optional arguments:
- `--method-index`: 0 for Random Testing, 1 for Fuzzer and 2 for MDPFuzz.
- `--method-seed`: random seed for the method. We used 2020, 2021, 2023.
- `--method-init`: sampling budget for the fuzzing techniques. Default is 1000.
- `--method-testing`: testing budget for the three methods. Note that it is automatically reduced by all previous executions (i.e., sampling budget for Fuzzer and sampling budget + coverage model initialization). Default is 5000.
- `--path`: path for logging. Because of relative path, add `../../` to be at the level of the repository. Default is `../../data_rq2/carla/`.

Finally, you can either launch an execution with or without GPU:
- CPU: `CUDA_VISIBLE_DEVICES= python SCRIPT.py ARGS METHOD_ARGS --disable-cuda --disable-cudnn`.
- GPU: `python SCRIPT.py ARGS METHOD_ARGS`. Note that we further needed to disable cudNN with the additional ` --disable-cudnn` argument.

### RQ2: Fault Discovery Evaluation

While we don't recommend to run the experiments in parallel (since their computational requirements), you can do so with the script `launch_rq2.sh`.
It launches the three methods with three seeds in CPU mode.
Otherwise, as the default configuration matches the experimental settings of RQ2 ($K=10$, $\tau=\gamma=0.01$, initial and testing budgets of 1000, 5000, respectively), the commands are the following:
```
args=--suite=town2 --max-run 100 --path-folder-model model_RL_IAs_only_town01_train_weather/ --crop-sky --seed 2022
# RT
CUDA_VISIBLE_DEVICES= python my_server_runner.py $args --disable-cuda --disable-cudnn --method-index 0 --method-seed 2021 --port=PORT1
CUDA_VISIBLE_DEVICES= python my_server_runner.py $args --disable-cuda --disable-cudnn --method-index 0 --method-seed 2022 --port=PORT2
CUDA_VISIBLE_DEVICES= python my_server_runner.py $args --disable-cuda --disable-cudnn --method-index 0 --method-seed 2023 --port=PORT3
# Fuzzer
CUDA_VISIBLE_DEVICES= python my_server_runner.py $args --disable-cuda --disable-cudnn --method-index 1 --method-seed 2021 --port=PORT4
CUDA_VISIBLE_DEVICES= python my_server_runner.py $args --disable-cuda --disable-cudnn --method-index 1 --method-seed 2022 --port=PORT5
CUDA_VISIBLE_DEVICES= python my_server_runner.py $args --disable-cuda --disable-cudnn --method-index 1 --method-seed 2023 --port=PORT6
# MDPFuzz
CUDA_VISIBLE_DEVICES= python my_server_runner.py $args --disable-cuda --disable-cudnn --method-index 2 --method-seed 2021 --port=PORT7
CUDA_VISIBLE_DEVICES= python my_server_runner.py $args --disable-cuda --disable-cudnn --method-index 2 --method-seed 2022 --port=PORT8
CUDA_VISIBLE_DEVICES= python my_server_runner.py $args --disable-cuda --disable-cudnn --method-index 2 --method-seed 2023 --port=PORT9
```

### RQ3: Parameter Analysis

MDPFuzz can be executed for a particular configuration and seed with the script `mdpfuzz.sh`.
It launches `my_server_runner.py` (in GPU mode), on port `$2` and with the configuration ($K$, $\tau$, $\gamma$) and seed read at line `$1` in the file `parameters.txt`.
