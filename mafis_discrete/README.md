## Installation

```bash
# venv
conda create -n MAFIS python=3.7 -y
conda activate MAFIS
pip install torch torchvision torchaudio
pip install sacred numpy scipy matplotlib seaborn pyyaml pygame pytest probscale imageio snakeviz tensorboard-logger tensorboard tensorboardx h5py
# SMACv2, Gold Miner, MPE
pip install -e smacv2
pip install -e miner/heuristic
pip install pettingzoo
pip install setuptools==65.5.0
pip install --user wheel==0.38.0
pip install gym==0.21.0
pip install importlib-metadata==4.13.0
bash install_sc2.sh
export SC2PATH=~/3rdparty/StarCraftII
```

## Run
Download expert demonstrations from [datasets](https://www.lamda.nju.edu.cn/lingzx/NeurIPS_2025_MAFIS_dataset.zip) to `./datasets/`. All the experiments can be run with the unified entrance file `src/main.py` with customized arguments.

Training scripts are also provided in the `runalgo.sh` script.

## Acknowledgement

+ [pymarl](https://github.com/oxwhirl/pymarl)