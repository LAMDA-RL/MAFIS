## Installation

```bash
# venv
# recommend using Python 3.7
uv venv
source .venv/bin/activate
uv pip install torch torchvision torchaudio
uv pip install sacred numpy scipy matplotlib seaborn pyyaml pygame pytest probscale imageio snakeviz tensorboard-logger tensorboard tensorboardx h5py
# SMACv2, Gold Miner, MPE
uv pip install -e smacv2
uv pip install -e miner/heuristic
uv pip install pettingzoo
uv pip install gym==0.22.0
uv pip install importlib-metadata==4.13.0
bash install_sc2.sh
export SC2PATH=~/3rdparty/StarCraftII
```

## Run
Download expert demonstrations from [datasets](https://www.lamda.nju.edu.cn/lingzx/NeurIPS_2025_MAFIS_dataset.zip) to `./datasets/`. All the experiments can be run with the unified entrance file `src/main.py` with customized arguments.

Training scripts are also provided in the `runalgo.sh` script.

## Acknowledgement

+ [pymarl](https://github.com/oxwhirl/pymarl)