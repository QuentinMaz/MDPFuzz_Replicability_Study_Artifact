# Use the ubuntu base image
FROM continuumio/miniconda3

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Don't cache pip packages, should reduce memory usage
ENV PIP_NO_CACHE_DIR=1

# Update and install dependencies
RUN apt-get update && \
    apt-get install -y wget bzip2 && \
    apt-get install -y g++ && \
    # apt-get install -y unzip && \
    apt-get install -y parallel && \
    apt-get clean

# Install Miniconda
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
#     /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
#     rm /tmp/miniconda.sh

# Add conda to the PATH
# ENV PATH=/opt/conda/bin:$PATH

COPY . /src

WORKDIR /src

# to enable bash commands
SHELL ["/bin/bash", "-c"]

# RUN conda init bash \
#     && . ~/.bashrc \
#     && conda create --name test-env python=3.7 \
#     && conda activate test-env \
#     && pip install ipython

# REPRODUCTION ENVS
## ACAS
RUN cd reproduction/ACAS_Xu && \
    conda create -n acas python=3.7.9 && \
    conda env update --name acas --file experiment_ACAS.yml && \
    conda install --name acas pandas && \
    cd ../..
## BW
# RUN conda init bash && . ~/.bashrc && \
#     cd reproduction/Bipedal_Walker && \
#     conda create -n RLWalk python=3.6.3 && \
#     conda activate RLWalk && \
#     conda env update --name RLWalk --file environment_RLWalk.yml && \
#     conda activate RLWalk && cp ./gym/setup.py ./ && \
#     pip install -e . && \
#     cp ./stable_baselines3/setup.py ./ && \
#     pip install -e . && \
#     cd ../..
## COOP
# RUN conda init bash && . ~/.bashrc && \
#     cd reproduction/Coop_Navi && \
#     conda create -n marl python=3.5.4 && \
#     conda env update --name marl --file commented_MARL.yml && \
#     conda activate marl && \
#     pip install --upgrade pip && \
#     pip install -r requirements.txt && \
#     pip install tensorflow-gpu==1.15.0 && \
#     pip install pandas==0.25.3 && \
#     cd ./maddpg && \
#     pip install -e . && \
#     cd ../multiagent-particle-envs && \
#     pip install -e . && \
#     cd ../../../
## CART
# RUN conda init bash && . ~/.bashrc && \
#     cd replication/cart && \
#     conda create -n cart python=3.10.12 && \
#     conda activate cart && \
#     pip install -r requirements.txt && \
#     pip install git+https://github.com/DennisGross/gimitest.git && \
#     cd ../..


## CARLA
# RUN conda init bash && . ~/.bashrc && \
#     cd replication/carla && \
#     wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.6.tar.gz && \
#     mkdir carla_RL_IAs && \
#     tar -xvzf CARLA_0.9.6.tar.gz -C carla_RL_IAs && \
#     cd carla_RL_IAs && \
#     wget http://www.cs.utexas.edu/~dchen/lbc_release/navmesh/Town01.bin && \
#     wget http://www.cs.utexas.edu/~dchen/lbc_release/navmesh/Town02.bin && \
#     mv Town*.bin CarlaUE4/Content/Carla/Maps/Nav/ && \
#     cd PythonAPI/carla/dist && \
#     rm carla-0.9.6-py3.5-linux-x86_64.egg && \
#     wget http://www.cs.utexas.edu/~dchen/lbc_release/egg/carla-0.9.6-py3.5-linux-x86_64.egg && \

#     conda create -n carla python=3.5.6 && \
#     conda activate carla && \
#     easy_install carla-0.9.6-py3.5-linux-x86_64.egg && \
#     conda deactivate && \
#     cd ../../../.. && \
#     # The installation warns that pip is not conda package.
#     # It additionally logs that numpy, scipy and pip were later replaced by the same version (1.15.2, 1.1.0 and 10.0.1)...
#     conda env update --name carla --file environment_carlarl.yml && \
#     conda activate carla && \

#     pip install numpy==1.18.5 && \
#     # only if you consider using GPU
#     # pip install torchvision torchaudio

#     wget https://github.com/marintoro/LearningByCheating/releases/download/v1.0/model_RL_IAs_only_town01_train_weather.zip && \
#     unzip model_RL_IAs_only_town01_train_weather.zip && \

#     # for "my_server_runner.py" script, see below
#     cp CarlaUE4_pid.sh carla_RL_IAs/CarlaUE4_pid.sh


# CMD ["bash", "-c", "source activate base && \
# conda init && \
# bash"]
CMD ["bash"]
# conda activate acas && \
# pip install matplotlib pandas && \
# python my_simulate_demo.py && \
# cp -r fuzzer_fuzzing_logs.txt /output && \
# cp -r mdpfuzz_fuzzing_logs.txt /output && \
# cp -r fault_discovery_plot.png /output"]
