import os
from os.path import dirname, exists
import glob
import json
from collections import OrderedDict as odict

####################################################
########          global configs           #########
####################################################

PROJPATH = f'{dirname(dirname(__file__))}'
results_dir = f'{PROJPATH}/results'
smry_tbls_dir = f'{PROJPATH}/summary'
paper_plots_dir = f'{PROJPATH}/paper_plots/main'
apdx_plots_dir = f'{PROJPATH}/paper_plots/appendix'
smry_fmt = 'h5'

#resdir_expname_method
a = [('1_ovat500m_west/1_ppo2_1s','ovat500m', 'ppo2'),
     ('1_ovat500m_west/2_ppo2_extra1s','ovat500m', 'ppo2'),
     ('1_ovat500m_west/3_ppo2_extra1s','ovat500m', 'ppo2'),
     ('1_ovat500m_west/4_ppo1_1s','ovat500m', 'ppo1'),
     ('1_ovat500m_west/4_trpo1s','ovat500m', 'trpo'),
     ('1_ovat500m_west/1_td3_1s','ovat500m', 'td3'),
     ('1_ovat500m_west/2_td3_1s','ovat500m2', 'td3'),
     ('2_ovat5b_eng/1_trpo','ovat5b', 'trpo'),
     ('2_ovat5b_eng/1_ppo1','ovat5b', 'ppo1'),
     ('2_ovat5b_eng/2_trpo','ovat5b', 'trpo'),
     ('2_ovat5b_eng/2_ppo1','ovat5b', 'ppo1'),
     ('2_ovat5b_eng/2_ppo2','ovat5b', 'ppo2'),
    ]

extra_tups_json = f'{PROJPATH}/configs/expnames.json'
if exists(extra_tups_json):
    with open(extra_tups_json) as f:
        extra_exptups = json.load(f, object_pairs_hook=odict)
    a = a + extra_exptups

# The frozen experiments are the ones whose original results and storage
# directories were lost in the great reset. These summary files are marked
# read-only in the file-system to prevent writing to them as well.
frozen_exps = ['ovat5b', 'ovat500m', 'ovateasy', 'hpoeasy',
               'ovatlowfreq', 'hpolowfreq', 'hpolowfreqeval',
               'hnhpolowfreq', 'lnhpolowfreq']

# making the translations
resdir_expname_method = []
for resdir_regex, expname, method in a:
    resdir_expname_method += [(resdirfp.replace(glob.glob(f'{results_dir}/')[0], ''), expname, method)
                              for resdirfp in glob.glob(f'{results_dir}/{resdir_regex}')]

"""
Summary of the experiments:
  * 1_ovat500m_west/1_ppo1_1s: This had a broken default entropy coefficient
        which caused the default hyper-parameters to have generate a reward of
        -1500 after training for 500 million steps. Other methods easily could
        generate a -1000 reward after 500M steps. After observing the ovat
        results, the only hyper-parameter that could push the reward to -1000
        was entcoeff=0.
  * '2_ovat5b_eng/1_ppo2': This setting has an unstable default. A few
        hyper-params could stabilize it, but I decided on setting ent_coef=0
        for stabilization for two reasons: (1) to be consistent with ppo1,
        (2) the other hyper-params that could stabilize it didn't yield a smooth
        changing trend (e.g., increasing/decreasing the hyper-param wasn't yielding
        a consistent trend towards improvement). The successor setting
        is '2_ovat5b_eng/2_ppo2'.

Broken/deprecated experiments:
  * ('1_ovat500m_west/1_trpo1s','ovat500m', 'trpo')
  * ('1_ovat500m_west/1_ppo1_1s','ovat500m', 'ppo1')
  * ('1_ovat500m_west/3_trpo_extra1s','ovat500m', 'trpo')
  * ('1_ovat500m_west/3_ppo1_extra1s','ovat500m', 'ppo1')
"""

##################################
####### Evaluation Columns #######
##################################
eval_xcols = ['eval_ntrajs', 'eval_nsteps', 'eval_seed']
eval_idcols = ['ckpt_name', 'eval_seed']
eval_ycols = ['traj_return', 'traj_nsteps', 'act_std']

##################################
########## TRPO Columns ##########
##################################
trpo_xcols = ['rng_seed', 'timesteps_per_batch', 'gamma',
              'lam', 'max_kl', 'cg_iters', 'entcoeff',
              'cg_damping', 'vf_stepsize', 'vf_iters',
              'vf_minibatches', 'method', 'environment',
              'total_timesteps', 'num_envs']

trpo_ycols = ['explained_variance_tdlam_before',
              'EpRewMean', 'entropy', 'entloss',
              'optimgain', 'meankl', 'surrgain']

trpo_tcols = ['TimestepsSoFar', 'TimeElapsed']

trpo_renames = dict()

##################################
########## PPO1 Columns ##########
##################################
ppo1_xcols = ['rng_seed', 'timesteps_per_actorbatch', 'gamma',
              'lam', 'clip_param', 'entcoeff', 'optim_epochs',
              'optim_stepsize', 'optim_minibatches', 'method',
              'adam_epsilon', 'schedule', 'environment',
              'total_timesteps', 'num_envs']

ppo1_tcols = ['TimestepsSoFar', 'TimeElapsed']

ppo1_ycols = ['loss_pol_surr','ev_tdlam_before', 'loss_ent', 'loss_kl',
              'EpRewMean', 'loss_vf_loss','loss_pol_entpen']

ppo1_renames = dict()

##################################
########## PPO2 Columns ##########
##################################
ppo2_xcols = ['rng_seed', 'gamma', 'lam', 'n_steps', 'ent_coef',
              'learning_rate', 'vf_coef', 'max_grad_norm',
              'nminibatches', 'noptepochs', 'cliprange',
              'method', 'environment', 'total_timesteps',
              'num_envs']

ppo2_tcols = ['total_timesteps', 'TimeElapsed', 'serial_timesteps',
              'time_elapsed', 'n_updates', 'TimestepsSoFar']

ppo2_ycols = ['explained_variance', 'fps', 'policy_loss', 'approxkl',
              'value_loss', 'policy_entropy', 'clipfrac', 'ep_len_mean',
              'ep_reward_mean']

ppo2_renames = {'total_timesteps': 'TimestepsSoFar'}

##################################
########## TD3 Columns ##########
##################################

td3_xcols = ['rng_seed', 'gamma', 'buffer_size', 'learning_starts',
             'train_freq', 'batch_size', 'learning_rate',
             'gradient_steps', 'tau', 'policy_delay',
             'actnoise_type', 'actnoise_freqrel', 'actnoise_stdrel',
             'target_policy_noise', 'target_noise_clip',
             'random_exploration', 'method', 'environment',
             'total_timesteps', 'num_envs']

td3_ycols = ['qf1_loss', 'qf2_loss', 'current_lr', 'fps', '100ep_rewmean']

td3_tcols = ['TimestepsSoFar', 'time_elapsed', 'n_updates']

td3_renames = {'total timesteps': 'TimestepsSoFar',
               'mean last 100 episodes reward': '100ep_rewmean',
               'eplenmean': 'EpLenMean',
               'total episodes': 'EpisodesSoFar',
               'ep_rewmean': 'All_ep_rewmean'}

##################################
######## Ignored Columns #########
##################################
ignore_cols= ['policy_kwargs', 'OPENAI_LOG_FORMAT', 'walltime_hrs',
              'num_checkpoints', 'results_dir', 'storage_dir',
              'train_set', 'eval_set', 'do_train', 'do_eval',
              'trash_storage_dir', 'train_done_path',
              'config_id',
              'eval_done_path', 'trash_results_dir',
              'EpisodesSoFar', 'EpLenMean', 'EpThisIter', 'All_ep_rewmean']

##################################
########## Column Dicts ##########
##################################
xcols = dict(trpo=trpo_xcols, ppo1=ppo1_xcols, ppo2=ppo2_xcols, td3=td3_xcols)
ycols = dict(trpo=trpo_ycols, ppo1=ppo1_ycols, ppo2=ppo2_ycols, td3=td3_ycols)
tcols = dict(trpo=trpo_tcols, ppo1=ppo1_tcols, ppo2=ppo2_tcols, td3=td3_tcols)
col_renames = dict(trpo=trpo_renames, ppo1=ppo1_renames, ppo2=ppo2_renames, td3=td3_renames)
evalcols = ['act_std', 'traj_nsteps', 'traj_return']

xcols = {method:list(set(xcol_lst)) for method, xcol_lst in xcols.items()}
ycols = {method:list(set(ycol_lst)) for method, ycol_lst in ycols.items()}
tcols = {method:list(set(tcol_lst)) for method, tcol_lst in tcols.items()}
