## Installation

```bash
# venv
# recommend using Python 3.7
uv venv
source .venv/bin/activate
uv pip install torch torchvision torchaudio
uv pip install sacred numpy scipy matplotlib seaborn pyyaml pygame pytest probscale imageio snakeviz tensorboard-logger tensorboard tensorboardx h5py
uv pip install gym==0.22.0
uv pip install importlib-metadata mujoco_py Cython jinja2
uv pip install -e .
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