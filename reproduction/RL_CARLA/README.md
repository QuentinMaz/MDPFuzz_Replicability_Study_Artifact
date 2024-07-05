# Reproduction of CARLA

## Installation

We remind the user that this use case cannot be executed inside the Docker container, and therefore the virtual environment needs to be installed on your local system.

We needed to slightly change the install process in order to install the Python virtual envrionment. While you can still try to follow the original instructions (detailed in `ORIGINAL_README.md`), we recommend doing the following instead:
```bash
# wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.6.tar.gz
# updated download link
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
```
The rest of this document assumes the environment to be activated.

## Experiments

First, open a terminal (remind to activate the carla environment) and run `./carla_RL_IAs/CarlaUE4.sh -fps=10 -benchmark -carla-port=PORT` to start the server.

Then, in a second terminal, run `python benchmark_agent.py --suite=town2 --max-run 100 --path-folder-model model_RL_IAs_only_town01_train_weather/ --crop-sky --port=PORT --seed SEED` with the same `PORT` value to start *Fuzzer-O*. Append to the previous command `--emguide` to launch *MDPFuzz-O* instead.
GPU can be manually disabled with the additional argument `--disable-cuda`.

Please note that:
- Since our hardware does not have GPU, we doubled the original sampling and fuzzing times. You can revert this change in `benchmark/run_benchmark.py` lines 183-184.
- We did not log the inputs tested. Indeed, they include the positions of all the vehicles (100) and we deemed the possibility the generating identical inputs negligible (let alone the stochastic nature of the executions).
- We used the following seeds: 2023, 2006 and 1453.

Logs of the executions are automatically saved in `../data/carla/`.

To summarize, replicating the paper can be done by running 3 times `python benchmark_agent.py ...` with and without `--emguide` (to execute *Fuzzer-O* and *MDPFuzz-O*) with the seeds 2006, 2023 and 1453. However, the executions being stochastic, the results won't be exactly the same as the ones presented in the paper.

#### Troubleshooting

- **Infinite loop:** when attempting to use Pytorch with CUDA enabled, loading the model (`benchmark_agent.py`, lines 38-40) creates an <ins>infinite loop</ins>. We solved the issue by propagating the Pytorch's device in the models' classes implemented in `bird_view/models/`. However, given the unstable nature of the implementation, we recommend disabling CUDA (`--disable-cuda`).