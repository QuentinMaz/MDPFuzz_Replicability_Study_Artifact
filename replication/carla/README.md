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

conda create -n carla2 python=3.5.6
conda activate carla2
easy_install carla-0.9.6-py3.5-linux-x86_64.egg
conda deactivate
cd ../../../..
# The installation warns that pip is not conda package.
# It additionally logs that numpy, scipy and pip were later replaced by the same version (1.15.2, 1.1.0 and 10.0.1)...
conda env update --name carla2 --file environment_carlarl.yml
conda activate carla2

pip install numpy==1.18.5
# only if you consider using GPU
pip install torchvision torchaudio

wget https://github.com/marintoro/LearningByCheating/releases/download/v1.0/model_RL_IAs_only_town01_train_weather.zip
unzip model_RL_IAs_only_town01_train_weather.zip

# for "my_server_runner.py" script, see below
cp CarlaUE4_pid.sh carla_RL_IAs/CarlaUE4_pid.sh
chmod +x carla_RL_IAs/CarlaUE4_pid.sh
```

If you are using the Docker image, make sure that the environment `carla2` is activated (`conda activate carla2`).

## Experiments

### RQ2: Fault Discovery Evaluation

Running one of the methods evaluated (Fuzzer, MDPFuzz and Random Testing) consists of a server and a client.
We provide two scripts, `my_runner.py` and `my_server_runner.py`, which execute a method with and without launching the server, respectively.

Therefore, when using `my_runner.py`, you need to open another terminal (remind to activate the environment `carla2`) and run `DISPLAY= ./carla_RL_IAs/CarlaUE4.sh -fps=10 -benchmark -carla-port=PORT -opengl` to start the server.
To avoid ports already in use, we recommend quite high `PORT` values (above 1500 and below 5000).

The scripts `my_runner.py` and `my_server_runner.py` share their argument, which are borrowed from the original implementation: `--suite=town2 --max-run 100 --path-folder-model model_RL_IAs_only_town01_train_weather/ --crop-sky --seed 2022 --port=PORT`.
The following additional arguments let you then select the method and configuration:
- `--method-index`: 0 for *Random Testing*, 1 for *Fuzzer-R* and 2 for *MDPFuzz-R*.
- `--method-seed`: random seed for the method. We used 2020, 2021, 2023.
- `--method-init`: sampling budget for the fuzzing techniques. In the paper, we use 1000 (default).
- `--method-testing`: testing budget for the three methods. Note that it is automatically reduced by all previous executions (i.e., sampling budget for Fuzzer and sampling budget + coverage model initialization for MDPFuzz). In the paper, we use 5000 (default).
- `--path`: path for logging. Because of relative path, add `../../` to be at the level of the repository. Default is `../../data_rq2/carla/`. To avoid any later issue when processing the results (see `../README`), we recommend not changing the default path.

Finally, you can either launch an execution with or without GPU:
- CPU: `CUDA_VISIBLE_DEVICES= python SCRIPT.py ARGS METHOD_ARGS --disable-cuda --disable-cudnn`.
- GPU: `python SCRIPT.py ARGS METHOD_ARGS`.

#### Notes

1. Our hardware could use GPU but without cudNN (i.e., we used the flag `--disable-cudnn`).
2. The scripts track the state of the execution differently. `my_runner.py` writes stdin in a `.txt` next to the log file of the results (whose default path is located in `../data_rq3/`). `my_server_runner.py`'s output is more complex, since we wanted to ensure that you will be able to kill the server process whatever happens (even though all the scenarios should be handled). Precisely, the script first writes the PID of the server process in `carlar_server_pid_{PORT}.txt`. Then, the state of the initialization of the method is written in `client_pid_{CLIENT_PID}.txt`. Once the server-client communication is established, stdin is redirected in a `.txt` file next to the results' logs (as `my_server_runner.py`). In case of any error during the execution of the method, the `my_server_runner.py` *should* eventually kill the server process. If not, read its PID in the file mentioned above with the command `kill {PID}`.
3. We suggest first trying short executions by setting low budget values with `--method-init` and `--method-testing`, e.g:
```python
args="--suite=town2 --max-run 100 --path-folder-model model_RL_IAs_only_town01_train_weather/ --crop-sky --seed 2022"
# RT with GPU
python my_server_runner.py $args --disable-cudnn --method-index 0 --method-seed 2021 --port=PORT1 --method-init 100 --method-testing 100
# or with CPU
CUDA_VISIBLE_DEVICES= python my_server_runner.py $args --disable-cuda --disable-cudnn --method-index 0 --method-seed 2021 --port=PORT1 --method-init 100 --method-testing 100
```
4. To replicate the paper, use the default settings and run the three methods with the seeds 2021, 2022 and 2023:
```python
args="--suite=town2 --max-run 100 --path-folder-model model_RL_IAs_only_town01_train_weather/ --crop-sky --seed 2022"
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

# remove "CUDA_VISIBLE_DEVICES= " and "--disable-cuda" to use CUDA
```

### RQ3: Parameter Analysis

MDPFuzz can be executed for a particular configuration and seed with the script `launch_mdpfuzz.sh`.
It launches `my_server_runner.py` (in GPU mode), on port `$2` and with the configuration ($K$, $\tau$, $\gamma$) and seed read at line `$1` in the file `parameters.txt`.

**To replicate the paper, you thus need to launch the script above 66 times, with $1 from 1 to 66**, e.g.:
```bash
./launch_mdpfuzz.sh 1 1453
./launch_mdpfuzz.sh 2 1500
./launch_mdpfuzz.sh 3 1550
# ...
```

#### Notes

1. You might need to make the script executable first with `chmod +x launch_mdpfuzz.sh`.