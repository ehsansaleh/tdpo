import pandas as pd
import os
from os.path import exists, isdir, isfile, abspath
from collections import defaultdict
import shutil
import sys

import argparse
my_parser = argparse.ArgumentParser()
my_parser.add_argument('-r', '--runbook_dir', action='store', type=str, required=True)
my_parser.add_argument('-s', '--size', action='store', type=int, required=True)
args = my_parser.parse_args()
args_runbook_dir = args.runbook_dir
args_size = args.size

rbdir = args_runbook_dir
world_size = args_size

if exists(rbdir) and isdir(rbdir):
    rbdir = abspath(rbdir)
    cfgid2int = lambda cfgid: int(cfgid.split('.')[0].split('_')[-1])
    csv_paths = [x for x in os.listdir(rbdir)
                 if (x.endswith('.csv') and isfile(f'{rbdir}/{x}'))]
else:
    csv_paths = []

print(f' -> Existing runbooks: {csv_paths}')

if len(csv_paths) < 1:
    # nothing to re-divide
    sys.exit(0)

csv_paths = sorted(csv_paths, key=cfgid2int)
csv_paths = [f'{rbdir}/{x}' for x in csv_paths]

full_rb = pd.concat([pd.read_csv(x) for x in csv_paths], axis=0, ignore_index=True)
full_rb['newrank'] = [(cfgid2int(x)%world_size) for x in full_rb['config_id'].tolist()]
rank_df_list = [full_rb[full_rb['newrank'] == rank] for rank in range(world_size)]

existing_round_dirs = [x for x in os.listdir(rbdir)
                       if (isdir(f'{rbdir}/{x}') and ('round_' in x))]
if len(existing_round_dirs) > 0:
    rounddir2int = lambda a: int(a.split('_')[-1])
    max_round = max([rounddir2int(x) for x in existing_round_dirs])
else:
    max_round = -1

new_rounddir = f'{rbdir}/round_{max_round + 1}'
os.makedirs(new_rounddir, exist_ok=True)
new_rounddir = abspath(new_rounddir)
for csv_path in csv_paths:
    shutil.move(csv_path, new_rounddir)

for rank, rnkdf in enumerate(rank_df_list):
    rnkdf_ = rnkdf.drop('newrank', 1)
    rnkdf_.to_csv(f'{rbdir}/runbook_{rank}.csv', index=False)
