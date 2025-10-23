#!/bin/bash
python3 src/main.py --config=MAFIS_online --env-config=sc2v2 with env_args.map_name="protoss_5_vs_5" epoch=5 alpha=0.5
python3 src/main.py --config=MAFIS_online --env-config=sc2v2 with env_args.map_name="protoss_10_vs_11" epoch=5 alpha=0.2
python3 src/main.py --config=MAFIS_online --env-config=sc2v2 with env_args.map_name="terran_5_vs_5" epoch=5 alpha=0.2
python3 src/main.py --config=MAFIS_online --env-config=sc2v2 with env_args.map_name="terran_10_vs_11" epoch=5 alpha=0.2
python3 src/main.py --config=MAFIS_online --env-config=sc2v2 with env_args.map_name="zerg_5_vs_5" epoch=5 alpha=0.5
python3 src/main.py --config=MAFIS_online --env-config=sc2v2 with env_args.map_name="zerg_10_vs_11" epoch=5 alpha=0.5
python3 src/main.py --config=MAFIS_online --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-spread-v3" reward_scalarisation="mean" epoch=5 alpha=0.2
python3 src/main.py --config=MAFIS_online --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-speaker-listener-v4" reward_scalarisation="mean" epoch=5 alpha=0.2
python3 src/main.py --config=MAFIS_online --env-config=gymma with env_args.time_limit=25 env_args.key="pz-mpe-simple-reference-v3" reward_scalarisation="mean" epoch=5 alpha=0.2
python3 src/main.py --config=MAFIS_online --env-config=miner with env_args.key="miner_easy_2_vs_2" epoch=2 alpha=0.2
python3 src/main.py --config=MAFIS_online --env-config=miner with env_args.key="miner_medium_2_vs_2" epoch=2 alpha=0.2
python3 src/main.py --config=MAFIS_online --env-config=miner with env_args.key="miner_hard_2_vs_2" epoch=2 alpha=0.2

