import sys
import argparse
import os
import copy
import gym
import numpy as np
import json
import warnings
import random
from mpi4py import MPI
from cleg.usr.binding import LegCRoller
from envs.leg import SimInterface, GymEnv
import torch
from xpo import xpo
from xpo.xpo import Explorer
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Simulate the environment with an agent')
parser.add_argument('-s', '--seed', type=int, default=0, help='experiment seed')
parser.add_argument('-e', '--env', type=str, default='', help='experiment env')
args = parser.parse_args()

myseed_ = 1234 * (args.seed + 1)
domain = args.env

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.size

##########################################################################################
# Fixing the seed. This will make the initial network parameters deterministically random!
random.seed(myseed_)
np.random.seed(myseed_)
torch.manual_seed(myseed_)

def net_init_f(name, p):
    if 'weight' in name:
        if 'mu' in name:
            torch.nn.init.xavier_uniform_(p, gain=.001)
        else:
            torch.nn.init.xavier_uniform_(p)
    elif 'bias' in name:
        torch.nn.init.constant_(p, 0.)

agent_specs = dict(actor_hidden_units=[64, 64], output_scale=1.,
                   actor_fixed_std=0., net_init_f=net_init_f,
                   actor_class='dnn')

hoizon_dict = {'StandingLegLH-v1': 4000,
               'StandingLegPDLH-v1': 4000,
               'StandingLegPDMH2-v1': 1000,
               'StandingLegPDMH1-v1': 500,
               'StandingLegPDSH-v1': 100}

horizon_ = hoizon_dict[domain]
act_x = 45

rew_x = 75.
C_w2 = 100. * rew_x / (act_x**2)
C_wg = 0.
max_w = 1000. * act_x
exploration_scale = 0.1 * act_x

train_options = dict()
hashed_seed = int.from_bytes([(myseed_//(10**(3-x)))%10 for x in range(4)], byteorder='little')
train_options['seed'] = hashed_seed
train_options['gamma'] = (1. - 1./horizon_)
train_options['number_of_steps'] = 40_000_000_000
train_options['steps_per_iteration_per_process'] = 2*horizon_
explorer = Explorer(init_std=exploration_scale * np.ones(2),
                    init_mu=None, lr = 1e-2)
vine_params = dict()
vine_params['samples_per_traj'] = 14
vine_params['d1'] = 10
vine_params['d2'] = 2
vine_params['exp_noise_generator'] = explorer
vine_params['method'] = 'finite_diff'
vine_params['efficient_reset'] = False
train_options['vine_params'] = vine_params
train_options['transfer_data_to_root'] = True
train_options['identity_stabilization'] = 2e0
train_options['alg'] = 'wtrpo'
train_options['verbose'] = False
ls_num = 9
a = np.log(0.00001)/(ls_num-2)
ls_coeffs = np.exp(a * np.arange(ls_num-1))
ls_coeffs = ls_coeffs.tolist() + [0.]
ls_coeffs = ls_coeffs * 24
wtrpo_opt_params = dict()
wtrpo_opt_params['type'] = 'cg'
wtrpo_opt_params['stop_criterion'] = ('iter', 10, 'eps', 1e-5)
wtrpo_opt_params['line_search_coeffs'] = ls_coeffs

wtrpo_opt_params['Cw2_Cwg'] = (C_w2, C_wg)
wtrpo_opt_params['max_w'] = max_w
train_options['wtrpo_opt_params'] = wtrpo_opt_params
train_options['walltime_hrs'] = 72
train_options['greedy_plot'] = True
train_options['noisy_plot'] = False
train_options['steps_per_snapshot'] = 20_000 * horizon_
train_options['log_prefix'] = domain
train_options['export_baseline_format'] = True
train_options['environment_type'] = 'roller'
opts_json_path = f'./envs/agent_interface_options.json'
pyxml_file = f'./envs/leg.xml'
libpath = f'./envs/libRollout.so'
with open(opts_json_path, 'r') as infile:
    interface_options = json.load(infile)
interface_options['xml_file'] = pyxml_file
interface_options['slurm_job_file'] = None
interface_metadata = {}

if mpi_rank == 0:
    sim_interface = SimInterface(options=interface_options,
                    metadata=interface_metadata)
    plot_env = GymEnv(sim_interface)
else:
    plot_env = None

def make_env():
    env_cpp = LegCRoller(lib_path=libpath, options=interface_options,
    plot_env=plot_env)
    return env_cpp

agent = xpo.PPOAgent(make_env, **agent_specs)
res = agent.train(train_options)
