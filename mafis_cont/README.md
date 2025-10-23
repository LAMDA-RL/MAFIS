## Installation

```bash
# venv
conda create -n MAFIS python=3.7 -y
conda activate MAFIS
pip install torch torchvision torchaudio
pip install sacred numpy scipy matplotlib seaborn pyyaml pygame pytest probscale imageio snakeviz tensorboard-logger tensorboard tensorboardx h5py
pip install setuptools==65.5.0
pip install --user wheel==0.38.0
pip install gym==0.21.0
pip install importlib-metadata==4.13.0
pip install -e .
pip install mujoco_py
pip install Cython==0.29.37
pip install jinja2
# MuJoCo
cd ~/
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
mkdir .mujoco
tar -xzvf mujoco210-linux-x86_64.tar.gz -C .mujoco/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

## Run
Download expert demonstations from [datesets](https://www.lamda.nju.edu.cn/lingzx/NeurIPS_2025_MAFIS_dataset.zip) to `examples/`. All the experiments can be run with the unified entrance file `examples/train.py` with customized arguments (Please modify `mafis/configs/envs_cfgs/mamujoco.yaml`).

Training scripts are also provided in the `examples/train.sh` script: 
```shell
cd examples
./train.sh
```

## Acknowledgement

+ [HARL](http://github.com/PKU-MARL/HARL)