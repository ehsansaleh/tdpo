import json
import os
import traceback
import sys
import shutil
import tarfile
import numpy as np
from os.path import abspath, exists
from collections import OrderedDict as odict
from itertools import product
from datetime import datetime
import pandas as pd

from utils.io_utils import cfg2hp, stdout_redirector, SBCallback
from utils.io_utils import scatter_ckpt_params_mpi
from utils.io_utils import scatter_dict_mpi_pkl
from utils.io_utils import gather_stats_mpi
from utils.io_utils import oldify_and_rename_file_if_exists
from utils.io_utils import RunbookManager
from utils.io_utils import should_train_eval
from utils.io_utils import initialize_mpi4py, unnecessary_wait
from utils.io_utils import process_policy_kwargs
from cleg.usr.binding import LegCRoller

def train_and_eval(setting_dict, mpi_tuple):
    mpi_comm, mpi_rank, mpi_size = mpi_tuple
    setting_dict_input = setting_dict # Just for json dump
    setting_dict = setting_dict.copy()
    config_id = setting_dict.pop('config_id')
    results_dir = setting_dict.pop('results_dir')
    storage_dir = setting_dict.pop('storage_dir')
    trash_results_dir = setting_dict.pop('trash_results_dir')
    trash_storage_dir = setting_dict.pop('trash_storage_dir')
    method = setting_dict.pop('method')
    env_name_type = setting_dict.pop('environment')
    OPENAI_LOG_FORMAT = setting_dict.pop('OPENAI_LOG_FORMAT')
    walltime_hrs = setting_dict.pop('walltime_hrs')
    num_checkpoints = setting_dict.pop('num_checkpoints')
    eval_ntrajs = setting_dict.pop('eval_ntrajs')
    eval_nsteps = setting_dict.pop('eval_nsteps')
    eval_seed = setting_dict.pop('eval_seed')
    if method == 'trpo':
        # initialization keyword arguments
        timesteps_per_batch = setting_dict.pop('timesteps_per_batch')
        gamma = setting_dict.pop('gamma')
        lam = setting_dict.pop('lam')
        max_kl = setting_dict.pop('max_kl')
        policy_kwargs = setting_dict.pop('policy_kwargs')
        cg_iters = setting_dict.pop('cg_iters')
        entcoeff = setting_dict.pop('entcoeff')
        cg_damping = setting_dict.pop('cg_damping')
        vf_stepsize = setting_dict.pop('vf_stepsize')
        vf_iters = setting_dict.pop('vf_iters')
        vf_minibatches = setting_dict.pop('vf_minibatches')
        vf_batchsize = max(1, timesteps_per_batch // vf_minibatches)
    elif method == 'ppo2':
        # initialization keyword arguments
        gamma = setting_dict.pop('gamma')
        lam = setting_dict.pop('lam')
        n_steps = setting_dict.pop('n_steps')
        ent_coef = setting_dict.pop('ent_coef')
        learning_rate = setting_dict.pop('learning_rate')
        vf_coef = setting_dict.pop('vf_coef')
        max_grad_norm = setting_dict.pop('max_grad_norm')
        nminibatches = setting_dict.pop('nminibatches')
        noptepochs = setting_dict.pop('noptepochs')
        cliprange = setting_dict.pop('cliprange')
        policy_kwargs = setting_dict.pop('policy_kwargs')
    elif method == 'ppo1':
        # initialization keyword arguments
        gamma = setting_dict.pop('gamma')
        lam = setting_dict.pop('lam')
        timesteps_per_actorbatch = setting_dict.pop('timesteps_per_actorbatch')
        clip_param = setting_dict.pop('clip_param')
        policy_kwargs = setting_dict.pop('policy_kwargs')
        entcoeff = setting_dict.pop('entcoeff')
        optim_epochs = setting_dict.pop('optim_epochs')
        optim_stepsize = setting_dict.pop('optim_stepsize')
        optim_minibatches = setting_dict.pop('optim_minibatches')
        optim_batchsize = max(1, timesteps_per_actorbatch // optim_minibatches)
        adam_epsilon = setting_dict.pop('adam_epsilon')
        schedule = setting_dict.pop('schedule')
    elif method == 'td3':
        gamma = setting_dict.pop('gamma')
        buffer_size = setting_dict.pop('buffer_size')
        learning_starts = setting_dict.pop('learning_starts')
        train_freq = setting_dict.pop('train_freq')
        batch_size = setting_dict.pop('batch_size')
        learning_rate = setting_dict.pop('learning_rate')
        gradient_steps = setting_dict.pop('gradient_steps')
        tau = setting_dict.pop('tau')
        policy_delay = setting_dict.pop('policy_delay')
        actnoise_type = setting_dict.pop('actnoise_type')
        actnoise_freqrel = setting_dict.pop('actnoise_freqrel')
        actnoise_stdrel = setting_dict.pop('actnoise_stdrel')
        target_policy_noise  = setting_dict.pop('target_policy_noise')
        target_noise_clip = setting_dict.pop('target_noise_clip')
        random_exploration = setting_dict.pop('random_exploration')
        policy_kwargs = setting_dict.pop('policy_kwargs')
    elif method == 'wtrpo':
        n_vine = setting_dict.pop('n_vine')
        gamma = setting_dict.pop('gamma')
        c_w2 = setting_dict.pop('c_w2')
        c_wg = setting_dict.pop('c_wg')
        explrn_init = setting_dict.pop('explrn_init')
        explrn_lr = setting_dict.pop('explrn_lr')
        explrn_steps = setting_dict.pop('explrn_steps')
        cg_damping = setting_dict.pop('cg_damping')
        cg_iters = setting_dict.pop('cg_iters')
        cg_eps = setting_dict.pop('cg_eps')
        ls_num = setting_dict.pop('ls_num')
        ls_span = setting_dict.pop('ls_span')
        ls_reps = setting_dict.pop('ls_reps')
        policy_kwargs = setting_dict.pop('policy_kwargs')
        # the following were not worth the coding trouble
        max_w = 4.5e4

    # other keyword arguments
    total_timesteps = setting_dict.pop('total_timesteps')
    rng_seed = setting_dict.pop('rng_seed')
    num_envs = setting_dict.pop('num_envs')
    train_done_path = setting_dict.pop('train_done_path')
    eval_done_path = setting_dict.pop('eval_done_path')
    do_train = setting_dict.pop('do_train')
    do_eval = setting_dict.pop('do_eval')

    # We have to move everything to the trash can!
    if mpi_rank == 0:
        if do_train and exists(results_dir):
            shutil.move(results_dir, trash_results_dir)
        if do_train and exists(storage_dir):
            shutil.move(storage_dir, trash_storage_dir)

    if mpi_comm is not None:
        mpi_comm.Barrier()

    unnecessary_wait(mpi_rank)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(storage_dir, exist_ok=True)
    if mpi_rank == 0:
        logging_path = f'{storage_dir}/stdouterr.txt'
        oldify_and_rename_file_if_exists(logging_path)
    else:
        logging_path = None

    if mpi_comm is not None:
        logging_path = mpi_comm.bcast(logging_path, root=0)

    open_vec_envs = []
    def pre_fnlzr():
        for vec_env_ in open_vec_envs:
            vec_env_.close()

    with stdout_redirector(logging_path, pre_finalizer=pre_fnlzr):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print('-' * 90)
        print('-' * 25 + f" Date and Time <--> {dt_string} " + '-' * 25)
        print('-' * 90)

        print('*'*20)
        for key,val in setting_dict_input.items():
            print(f'{key}: {val}')
        print('*'*20)

        if len(setting_dict) > 0:
            msg_  = f'The following settings were left unused:\n'
            for key in setting_dict:
                msg_ += f'  {key}: {setting_dict[key]}'
            raise RuntimeError(msg_)

        if method in ('trpo', 'ppo1', 'td3', 'wtrpo'):
            msg_ = f'{mpi_size} == mpi_size !=  num_envs == {num_envs}'
            assert mpi_size == num_envs, msg_
        elif method in ('ppo2',):
            msg_ = f'for {method} mpi_size should be 1, not {mpi_size}.'
            assert mpi_size == 1, msg_
        else:
            raise ValueError(f'Not implemented for method {method}')

        ########################################
        ##### Dumping the Hyper-Parameters #####
        ########################################
        unnecessary_wait(mpi_rank)
        for config_path in (f'{storage_dir}/config.json',
                            f'{results_dir}/config.json'):
            if not exists(config_path):
                with open(config_path, 'w') as fp:
                    json.dump(setting_dict_input, fp, indent=4)

        ########################################
        ####### Creating the Environment #######
        ########################################
        pkg = process_policy_kwargs(policy_kwargs, method)
        policy_kwargs_sb, roller_mlpkwargs = pkg

        activation = roller_mlpkwargs['activation']
        do_mlp_output_tanh = roller_mlpkwargs['do_mlp_output_tanh']
        mlp_output_scaling = roller_mlpkwargs['mlp_output_scaling']
        h1 = roller_mlpkwargs['h1']
        h2 = roller_mlpkwargs['h2']

        is_env_roller = '_stepper' not in env_name_type
        env_type = 'roller' if is_env_roller else 'stepper'
        env_name = env_name_type.replace('_stepper', '')

        if env_name in ('drop_leg_4k', 'drop_leg_100hz',
                        'drop_leg_100hz_easy', 'drop_leg_100hz_easy_nc'):
            opts_json_path = f'../envs/agent_interface_options.json'
            pyxml_file = f'../envs/leg.xml'

            use_td3_mlp = ((activation == 'relu') and
                           (do_mlp_output_tanh == True) and
                           (mlp_output_scaling == 10) and
                           (h1 == 64) and (h2==64))

            use_xpo_mlp = ((activation == 'tanh') and
                           (do_mlp_output_tanh == False) and
                           (mlp_output_scaling == 1) and
                           (h1 == 64) and (h2==64))

            if (env_name, use_xpo_mlp) == ('drop_leg_4k', True):
                libpath = f'../envs/libRollout.so'
            elif (env_name, use_td3_mlp) == ('drop_leg_4k', True):
                libpath = f'../envs/libRollout_td3mlp.so'
            elif (env_name, use_xpo_mlp) == ('drop_leg_100hz', True):
                libpath = f'../envs/libRollout_100hz_xpo.so'
            elif (env_name, use_td3_mlp) == ('drop_leg_100hz', True):
                libpath = f'../envs/libRollout_100hz_td3mlp.so'
            elif (env_name, use_xpo_mlp) == ('drop_leg_100hz_easy', True):
                libpath = f'../envs/libRollout_100hz_xpo_easy.so'
            elif (env_name, use_td3_mlp) == ('drop_leg_100hz_easy_nc', True):
                libpath = f'../envs/libRollout_100hz_td3mlp_easy_nc.so'
            else:
                raise RuntimeError(f'binary does not exist for such mlp configs')

            with open(opts_json_path, 'r') as infile:
                interface_options = json.load(infile)
            interface_options['xml_file'] = pyxml_file
            interface_options['slurm_job_file'] = None
            interface_options['reward_scaling'] = 0.001
            interface_options['activation'] = activation
            interface_options['do_mlp_output_tanh'] = do_mlp_output_tanh
            interface_options['mlp_output_scaling'] = mlp_output_scaling
            interface_options['h1'] = h1
            interface_options['h2'] = h2

            if env_name == 'drop_leg_100hz':
                interface_options['outer_loop_rate'] = 100
            elif env_name in ('drop_leg_100hz_easy', 'drop_leg_100hz_easy_nc'):
                interface_options['outer_loop_rate'] = 100
                interface_options['inner_loop_rate'] = 400
                interface_options['omega_hip_init'] = [0, 0]
                interface_options['omega_knee_init'] = [0, 0]
                interface_options['vel_slider_init'] = [0., 0.]
                interface_options['theta_hip_init'] = [-0.8726, -0.8726]
                interface_options['theta_knee_init'] = [-1.7453, -1.7453]
                interface_options['pos_slider_init'] = [0.4, 0.4]
                interface_options['do_obs_noise'] = False
                interface_options['stand_reward'] = 'simplified'
                if env_name  == 'drop_leg_100hz_easy_nc':
                    interface_options['ground_contact_type'] = 'noncompliant'

            # if method == 'td3':
            #     interface_options['reward_scaling'] *= 100

            interface_metadata = {}

            if not is_env_roller:
                sys.path.insert(-1, abspath('../'))
                from envs.leg import SimInterface, GymEnv

            def stepper_env_maker_fn():
                my_intfce_opts = dict(interface_options)
                for key in ['do_mlp_output_tanh', 'h1', 'h2',
                            'activation', 'mlp_output_scaling']:
                    my_intfce_opts.pop(key, None)
                sim_interface = SimInterface(options=my_intfce_opts,
                                 metadata=interface_metadata)
                env = GymEnv(sim_interface)
                def env_eval_ckpts(*args, **kwargs):
                    assert hasattr(env, 'my_ckpt_params_dict')
                    my_ckpt_params_dict = env.my_ckpt_params_dict
                    eval_stats = eval_ckpts_stepper(env, my_ckpt_params_dict,
                                                    *args, **kwargs)
                    return eval_stats
                env.eval_ckpts = env_eval_ckpts
                return env

            def roller_env_maker_fn():
                env = LegCRoller(lib_path=libpath, options=interface_options)
                def env_eval_ckpts(*args, **kwargs):
                    assert hasattr(env, 'my_ckpt_params_dict')
                    my_ckpt_params_dict = env.my_ckpt_params_dict
                    eval_stats = eval_ckpts_roller(env, my_ckpt_params_dict,
                                                   *args, **kwargs)
                    return eval_stats
                env.eval_ckpts = env_eval_ckpts
                return env

            env_maker_fn = roller_env_maker_fn if is_env_roller else stepper_env_maker_fn

            max_traj_steps = int(interface_options['outer_loop_rate'] *
                                 interface_options['time_before_reset'])
            act_dim = 2
        elif env_name in ('Pendulum-v0', 'Pendulum1k-v0', 'Pendulum10k-v0'):
            assert not is_env_roller

            from gym.envs.registration import register
            register(id='Pendulum1k-v0',
                     entry_point='utils.pend_envs:PendulumEnv1K',
                     max_episode_steps=2000)
            register(id='Pendulum10k-v0',
                     entry_point='utils.pend_envs:PendulumEnv10K',
                     max_episode_steps=20000)

            max_traj_steps = {'Pendulum-v0':200,
                              'Pendulum1k-v0':2000,
                              'Pendulum10k-v0':20000}[env_name]
            act_dim = 1
            def env_maker_fn():
                env = make_vec_env(env_name, n_envs=1)
                def env_eval_ckpts(*args, **kwargs):
                    assert hasattr(env, 'my_ckpt_params_dict')
                    my_ckpt_params_dict = env.my_ckpt_params_dict
                    eval_stats = eval_ckpts_stepper(env, my_ckpt_params_dict,
                                                    *args, **kwargs)
                    return eval_stats
                def env_seed(seed):
                    seeds = list()
                    for idx, myenv in enumerate(env.envs):
                        seeds.append(myenv.seed(seed + idx))
                    return seeds
                env.seed = env_seed
                env.eval_ckpts = env_eval_ckpts
                env.get_timestep = lambda : env.envs[0].dt
                return env
        else:
            raise RuntimeError(f'Unknown environment name {env_name}')

        ########################################
        ##### Configuring stable_baselines #####
        ########################################
        os.environ['OPENAI_LOG_FORMAT'] = OPENAI_LOG_FORMAT
        os.environ['OPENAI_LOG_FORMAT_MPI'] = ''
        os.environ['OPENAI_LOGDIR'] = results_dir

        unnecessary_wait(mpi_rank)
        from utils.sb_utils import BetterMlpPolicy, BetterSubprocVecEnv
        from utils.sb_utils import patch_ppo2, patch_trpo_ppo1
        from utils.sb_utils import get_tf_actor_weights
        unnecessary_wait(mpi_rank)

        if method in ('trpo', 'ppo1'):
            #assert is_env_roller
            if not is_env_roller:
                from stable_baselines.common import make_vec_env
            from stable_baselines import PPO1, TRPO
            patch_trpo_ppo1()
        elif method in ('ppo2',):
            assert is_env_roller
            from stable_baselines import PPO2
            patch_ppo2()
        elif method in ('td3',):
            from stable_baselines.td3.policies import MlpPolicy as TD3MlpPolicy
            from utils.sb_utils import UnlimitedTD3Policy
            if is_env_roller:
                from utils.td3_roller import TD3ME as TD3
                from utils.td3_roller import NoiseGenerator
            else:
                from stable_baselines import TD3MPIME as TD3
                from stable_baselines.common.noise import (NormalActionNoise,
                  OrnsteinUhlenbeckActionNoise)
                from stable_baselines.common import make_vec_env
        elif method in ('wtrpo',):
            from utils import xpo
            from utils.xpo import Explorer
        from stable_baselines.common.policies import MlpPolicy
        from stable_baselines.logger import configure as configure_sb_logger

        unnecessary_wait(mpi_rank)
        if do_train:
            configure_sb_logger()

        tar_path = f'{storage_dir}/checkpoints.tar'
        sb_callback = None
        if method in ('trpo', 'ppo1', 'ppo2', 'td3'):
            sb_callback = SBCallback(method, walltime_hrs, total_timesteps,
                                     num_checkpoints, tar_path, mpi_rank,
                                     mpi_size, mpi_comm)

        if 'act_fun' in policy_kwargs_sb:
            import tensorflow as tf
            s2f = {'tanh': tf.tanh, 'relu': tf.nn.relu}
            policy_kwargs_sb['act_fun'] = s2f[policy_kwargs_sb['act_fun']]

        ########################################
        ######## Constructing the Model ########
        ########################################
        noise_generator = None
        if method == 'trpo':
            #assert is_env_roller
            unnecessary_wait(mpi_rank)
            env = env_maker_fn()
            my_seed = 12345 + mpi_rank * 5 + rng_seed * 1000
            model = TRPO(MlpPolicy, env, seed=my_seed,
                         timesteps_per_batch=timesteps_per_batch,
                         gamma=gamma, lam=lam, max_kl=max_kl,
                         policy_kwargs=policy_kwargs_sb, cg_iters=cg_iters,
                         entcoeff=entcoeff, cg_damping=cg_damping,
                         vf_stepsize=vf_stepsize, vf_iters=vf_iters,
                         vf_batchsize=vf_batchsize, verbose=True,
                         env_type=env_type)
        elif method == 'ppo1':
            assert is_env_roller
            unnecessary_wait(mpi_rank)
            env = env_maker_fn()
            my_seed = 12345 + mpi_rank * 5 + rng_seed * 1000
            model = PPO1(MlpPolicy, env, seed=my_seed,
                         gamma=gamma, lam=lam, clip_param=clip_param,
                         timesteps_per_actorbatch=timesteps_per_actorbatch,
                         policy_kwargs=policy_kwargs_sb, entcoeff=entcoeff,
                         optim_epochs=optim_epochs, adam_epsilon=adam_epsilon,
                         optim_stepsize=optim_stepsize, schedule=schedule,
                         optim_batchsize=optim_batchsize, verbose=True)
        elif method == 'ppo2':
            assert is_env_roller
            unnecessary_wait(mpi_rank)
            vec_env = BetterSubprocVecEnv([env_maker_fn] * num_envs, start_method=None)
            open_vec_envs.append(vec_env)
            my_seed = 12345 + rng_seed * 1000
            model = PPO2(BetterMlpPolicy, vec_env, seed=my_seed,
                         gamma=gamma, lam=lam, n_steps=n_steps,
                         policy_kwargs=policy_kwargs_sb, ent_coef=ent_coef,
                         learning_rate=learning_rate, vf_coef=vf_coef,
                         max_grad_norm=max_grad_norm, nminibatches=nminibatches,
                         noptepochs=noptepochs, cliprange=cliprange, verbose=True)
        elif (method == 'td3') and (is_env_roller):
            unnecessary_wait(mpi_rank)
            env = env_maker_fn()
            my_seed = 12345 + mpi_rank * 5 + rng_seed * 1000
            env_dt =  env.get_timestep()
            env_actscale = (env.action_space.high - env.action_space.low) / 2.

            rng_props = dict()
            rng_props['type'] = actnoise_type
            rng_props['timestep'] = env_dt
            rng_props['rolloff_hz'] = actnoise_freqrel * np.pi / env_dt
            rng_props['output_coeff'] = actnoise_stdrel * env_actscale
            rng_props['steady_state_initialization'] = True
            noise_generator = NoiseGenerator(action_dim=env.action_dim, rng_props=rng_props)

            use_td3_mlp = ((activation == 'relu') and
                           (do_mlp_output_tanh == True) and
                           (mlp_output_scaling == 10) and
                           (h1 == 64) and (h2==64))

            if use_td3_mlp:
                TD3PolicyClass = TD3MlpPolicy
            else:
                TD3PolicyClass = UnlimitedTD3Policy
                policy_kwargs_sb['do_mlp_output_tanh'] = do_mlp_output_tanh
                policy_kwargs_sb['mlp_output_scaling'] = mlp_output_scaling

            model = TD3(TD3PolicyClass, env, seed=my_seed,
                        gamma=gamma, buffer_size=buffer_size,
                        learning_starts=learning_starts,
                        train_freq=train_freq, batch_size=batch_size,
                        learning_rate=learning_rate,
                        gradient_steps=gradient_steps,
                        tau=tau, policy_delay=policy_delay,
                        target_policy_noise=target_policy_noise,
                        target_noise_clip=target_noise_clip,
                        random_exploration=random_exploration,
                        action_noise=noise_generator,
                        policy_kwargs=policy_kwargs_sb,
                        verbose=True)

            # td3paramsdict = model.get_parameters()
            # td3paramsdict['model/pi/dense/kernel:0'] *= 0.001
            # td3paramsdict['target/pi/dense/kernel:0'] *= 0.001
            # td3paramsdict['model/pi/dense/bias:0'] *= 0.001
            # td3paramsdict['target/pi/dense/bias:0'] *= 0.001
            # model.load_parameters(td3paramsdict)
        elif (method == 'td3'):
            unnecessary_wait(mpi_rank)
            env = env_maker_fn()
            my_seed = 12345 + mpi_rank * 5 + rng_seed * 1000
            env_dt =  env.get_timestep()
            env_actscale = (env.action_space.high - env.action_space.low) / 2.

            if is_env_roller:
                rng_props = dict()
                rng_props['type'] = actnoise_type
                rng_props['timestep'] = env_dt
                rng_props['rolloff_hz'] = actnoise_freqrel * np.pi / env_dt
                rng_props['output_coeff'] = actnoise_stdrel * env_actscale
                rng_props['steady_state_initialization'] = True
                noise_generator = NoiseGenerator(action_dim=env.action_dim,
                                                 rng_props=rng_props)
            else:
                n_actions = env.action_space.shape[-1]
                noise_mean = np.zeros(n_actions)
                if actnoise_type == 'normal':
                    noise_generator = NormalActionNoise(mean=noise_mean,
                                                        sigma=actnoise_stdrel)
                elif actnoise_type == 'ornstein':
                    theta = actnoise_freqrel / env_dt # i.e., rolloff_hz / np.pi
                    sigma = np.sqrt(theta * (2.0 - theta * env_dt)) # This gives unit variance
                    sigma = sigma * actnoise_stdrel
                    noise_generator = OrnsteinUhlenbeckActionNoise(mean=noise_mean,
                        sigma=sigma, theta=theta, dt=env_dt, initial_noise=None)
                else:
                    raise RuntimeError(f'noise {actnoise_type} not implemented.')

            use_td3_mlp = ((activation == 'relu') and
                           (do_mlp_output_tanh == True) and
                           (mlp_output_scaling == 10) and
                           (h1 == 64) and (h2==64))
            if use_td3_mlp:
                TD3PolicyClass = TD3MlpPolicy
            else:
                TD3PolicyClass = UnlimitedTD3Policy
                policy_kwargs_sb['do_mlp_output_tanh'] = do_mlp_output_tanh
                policy_kwargs_sb['mlp_output_scaling'] = mlp_output_scaling

            model = TD3(TD3PolicyClass, env, seed=my_seed,
                        gamma=gamma, buffer_size=buffer_size,
                        learning_starts=learning_starts,
                        train_freq=train_freq, batch_size=batch_size,
                        learning_rate=learning_rate,
                        gradient_steps=gradient_steps,
                        tau=tau, policy_delay=policy_delay,
                        target_policy_noise=target_policy_noise,
                        target_noise_clip=target_noise_clip,
                        random_exploration=random_exploration,
                        action_noise=noise_generator,
                        policy_kwargs=policy_kwargs_sb,
                        verbose=True)

            # td3paramsdict = model.get_parameters()
            # td3paramsdict['model/pi/dense/kernel:0'] *= 0.001
            # td3paramsdict['target/pi/dense/kernel:0'] *= 0.001
            # td3paramsdict['model/pi/dense/bias:0'] *= 0.001
            # td3paramsdict['target/pi/dense/bias:0'] *= 0.001
            # model.load_parameters(td3paramsdict)
        elif method == 'wtrpo':
            assert is_env_roller
            unnecessary_wait(mpi_rank)
            my_seed = 1234 + rng_seed * 1000 # XPO will account for mpi_rank

            env = env_maker_fn()

            ####################################################################
            # Fixing the seed. Important for controlled MLP parameter init.
            import random, torch
            random.seed(my_seed)
            np.random.seed(my_seed)
            torch.manual_seed(my_seed)
            hashed_seed = [(my_seed//(10**(3-x)))%10 for x in range(4)]
            hashed_seed = int.from_bytes(hashed_seed, byteorder='little')

            aa = np.log(ls_span)/(ls_num-2)
            ls_coeffs = np.exp(aa * np.arange(ls_num-1))
            ls_coeffs = ls_coeffs.tolist() + [0.]
            ls_coeffs = ls_coeffs * ls_reps
            cg_stop_cri = ('iter', cg_iters, 'eps', cg_eps)
            explorer = Explorer(init_std=explrn_init*np.ones(act_dim),
                                init_mu=None, lr=explrn_lr)
            vine_params = dict(n_vine_samples=n_vine*2, d1=explrn_steps, d2=2,
                               exp_noise_generator=explorer, method='finite_diff',
                               efficient_reset=False)
            wtrpo_opt_params = dict(type='cg', C_w2=c_w2, C_wg=c_wg,
                                    max_w=max_w, stop_criterion=cg_stop_cri,
                                    line_search_coeffs=ls_coeffs)

            train_options = dict(method=method, total_timesteps=total_timesteps,
                                 seed=hashed_seed, gamma=gamma,
                                 steps_per_iteration_per_process=max_traj_steps,
                                 vine_params=vine_params,
                                 wtrpo_opt_params=wtrpo_opt_params,
                                 identity_stabilization=cg_damping,
                                 transfer_data_to_root=True, verbose=False,
                                 walltime_hrs=walltime_hrs, greedy_plot=False,
                                 noisy_plot=False, log_prefix='',
                                 logdir_base = storage_dir, logdir_tail = '',
                                 export_baseline_format=True,
                                 environment_type='roller',
                                 number_of_snapshots=num_checkpoints)

            model = xpo.PPOAgent(env, **policy_kwargs)
        else:
            raise RuntimeError(f'method not implemented: {method}')

        ########################################
        ########## Training the Model ##########
        ########################################
        train_start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        train_exc_tb = None
        had_train_exc = False
        if do_train:
            try:
                unnecessary_wait(mpi_rank)
                if method in ('trpo', 'ppo1', 'ppo2', 'td3'):
                    model.learn(total_timesteps=total_timesteps,
                                callback=sb_callback)
                elif method in ('wtrpo',):
                    res = model.learn(train_options)
                else:
                    raise RuntimeError(f'{method} not implemented')
            except Exception as train_exc:
                had_train_exc = True
                exc_type, exc_value, exc_traceback = sys.exc_info()
                train_exc_tb = traceback.format_exception(exc_type, exc_value,
                                                          exc_traceback,
                                                          limit=None, chain=True)
                print(''.join(train_exc_tb), flush=True)
        train_finish_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        if sb_callback is not None:
            sb_callback.close()
        else:
            model.close() # for xpo

        if do_train and (mpi_rank == 0):
            done_info = dict()
            if sb_callback is not None:
                done_info = sb_callback.get_completion_info()
            done_info['train_start_time'] = train_start_time
            done_info['train_finish_time'] = train_finish_time
            done_info['had_exception'] = had_train_exc
            done_info['exception_tb'] = train_exc_tb
            with open(train_done_path, 'w') as fp:
                json.dump(done_info, fp, indent=4)

        ########################################
        ###### Evaluating the Checkpoints ######
        ########################################
        if had_train_exc:
            return False

        if not do_eval:
            return True

        if not exists(tar_path):
            print(f'No tar at {tar_path}. Evaluation is cancelled.')
            return False

        ########################################
        ### Loading & Scattering Checkpoints ###
        ########################################
        ckpt_params_dict = None
        if mpi_rank == 0:
            ckpt_params_dict = odict()
            archive = tarfile.open(tar_path, 'r:')
            for member in archive.getmembers():
                fp = archive.extractfile(member)
                if method == 'wtrpo':
                    model.load(fp)
                    a = dict(fc1 = model.actor.fc1.weight.data.numpy(),
                             fc2 = model.actor.fc2.weight.data.numpy(),
                             fc3 = model.actor.fc3_mu.weight.data.numpy(),
                             fc1_bias = model.actor.fc1.bias.data.numpy(),
                             fc2_bias = model.actor.fc2.bias.data.numpy(),
                             fc3_bias = model.actor.fc3_mu.bias.data.numpy(),
                             policy_logstd = None)
                    named_parameters = a
                elif method in ('trpo', 'ppo1', 'ppo2', 'td3'):
                    model.load_parameters(fp)
                    if is_env_roller:
                        named_parameters = get_tf_actor_weights(model.sess, naming=method)
                    else:
                        named_parameters = model.get_parameters()
                ckpt_params_dict[member.name] = named_parameters
            archive.close()

        if method in ('trpo', 'ppo1', 'td3', 'wtrpo'):
            my_ckpt_params_dict = scatter_dict_mpi_pkl(ckpt_params_dict, mpi_comm,
                                                       mpi_rank, mpi_size)
        elif method in ('ppo2'):
            w = [dict() for _ in range(num_envs)]
            for i, (zip_name, named_params) in enumerate(ckpt_params_dict.items()):
                w[i % num_envs][zip_name] = named_params
            for j in range(num_envs):
                vec_env.set_attr('my_ckpt_params_dict', w[j], indices=[j])
        else:
            raise ValueError('Not implemented')

        ########################################
        ###### Performing the Evaluations ######
        ########################################
        eval_start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        if method in ('trpo', 'ppo1', 'ppo2'):
            eval_noise_type = 'policy_logstd'
        elif method in ('td3',):
            eval_noise_type = 'noise_generator'
        elif method in ('wtrpo',):
            eval_noise_type = 'zeros'
        else:
            raise ValueError(f'{method} not implemented')

        assert eval_noise_type in ('policy_logstd', 'noise_generator', 'zeros')

        eval_args = [eval_ntrajs, eval_nsteps, eval_seed]
        if is_env_roller:
            eval_args = eval_args + [eval_noise_type, noise_generator]
            eval_ckpts = eval_ckpts_roller
        else:
            determinisim = (eval_noise_type == 'zeros')
            eval_args = eval_args + [determinisim, model, method]
            eval_ckpts = eval_ckpts_stepper

        if method in ('trpo', 'ppo1', 'td3', 'wtrpo'):
            eval_stats, eval_excs = eval_ckpts(env, my_ckpt_params_dict, *eval_args)
            agg_eval_stats = gather_stats_mpi(eval_stats, mpi_comm, mpi_rank, mpi_size)
            agg_eval_excs_lst = gather_stats_mpi(eval_excs, mpi_comm, mpi_rank, mpi_size)
            agg_eval_excs = dict(agg_eval_excs_lst)
        elif method in ('ppo2'):
            eval_outputs = vec_env.env_method('eval_ckpts', *eval_args)
            agg_eval_stats = []
            agg_eval_excs = dict()
            for pkg in eval_outputs:
                ev_st_list, ev_ex_dict = pkg
                agg_eval_stats = agg_eval_stats + ev_st_list
                agg_eval_excs.update(ev_ex_dict)
        else:
            raise ValueError('Not implemented')

        eval_finish_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        ########################################
        #### Storing the Evaluation Reults #####
        ########################################
        eval_csv_path = f'{results_dir}/eval.csv'
        if mpi_rank == 0:
            df = pd.DataFrame(agg_eval_stats)
            if 'ckpt_name' in df.columns:
                df['ckpt_numsamps'] = [int(x.split('.')[0].split('_')[-1])
                                       for x in df['ckpt_name']]
                sort_cols = ['ckpt_numsamps', 'ckpt_name', 'eval_seed']
                if set(sort_cols).issubset(df.columns):
                    df = df.sort_values(by=sort_cols)
                df = df.drop('ckpt_numsamps', 1)
            df.to_csv(eval_csv_path, index=False)

            with open(eval_done_path, 'w') as fp:
                done_info = dict()
                done_info['eval_start_time'] = eval_start_time
                done_info['eval_finish_time'] = eval_finish_time
                done_info['had_exception'] = (len(agg_eval_excs) > 0)
                done_info['exception_tbs'] = agg_eval_excs
                json.dump(done_info, fp, indent=4)

        return True

    return False

def eval_ckpts_roller(env, my_ckpt_params_dict, eval_ntrajs, eval_nsteps,
                      eval_seed, noise_type, noise_generator):
    all_my_eval_stats = []
    all_my_excs_list = []
    for zip_name, named_params in my_ckpt_params_dict.items():
        try:
            policy_logstd = named_params.pop('policy_logstd')
            env.set_np_policy(named_params)
            env.seed(eval_seed) # This makes sure all actors get the same initializations

            assert noise_type in ('policy_logstd', 'noise_generator', 'zeros')

            if noise_type == 'noise_generator':
                noise_generator.seed(eval_seed)
                expl_noise = np.zeros((eval_ntrajs, eval_nsteps, env.action_dim),
                                      dtype=np.float64, order='C')
                for ii in range(eval_ntrajs):
                    noise_generator.reset()
                    expl_noise[ii] = noise_generator(eval_nsteps)
                act_std = noise_generator.get_std()
            elif noise_type == 'policy_logstd':
                # We're probably dealing with trpo/ppo1/ppo2
                policy_std = np.exp(policy_logstd).reshape(1, 1, env.action_dim)
                expl_noise = env.np_random.randn(eval_ntrajs, eval_nsteps, env.action_dim) * policy_std
                act_std = np.sqrt(np.square(policy_std).sum()).item()
            elif noise_type == 'zeros':
                # We're probably dealing with td3
                policy_std = np.zeros((1, 1, env.action_dim))
                expl_noise = env.np_random.randn(eval_ntrajs, eval_nsteps, env.action_dim) * policy_std
                act_std = np.sqrt(np.square(policy_std).sum()).item()
            else:
                raise ValueError(f'{noise_type} not implemented')

            out_dict = env.stochastic(eval_ntrajs, eval_nsteps, expl_noise=expl_noise)

            observations = out_dict['observations'] # (traj_num, n_steps, obs_dim)
            actions = out_dict['actions'] # (traj_num, n_steps, act_dim)
            rewards = out_dict['rewards'] # (traj_num, n_steps)
            done_steps = out_dict['done_steps'] # (traj_num,)

            returns = rewards.sum(axis=1)
            curr_stats = []
            for traj_idx in range(eval_ntrajs):
                traj_return = returns[traj_idx].item()
                traj_nsteps = done_steps[traj_idx].item()
                traj_seed = f'{eval_seed}_{traj_idx}'
                stat_dict = dict(ckpt_name=zip_name, traj_return=traj_return,
                                 eval_seed=traj_seed, traj_nsteps=traj_nsteps,
                                 act_std=act_std)
                curr_stats.append(stat_dict)
        except Exception as eval_exc:
            curr_stats = []
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_text = traceback.format_exception(exc_type, exc_value,
                                                 exc_traceback,
                                                 limit=None, chain=True)
            all_my_excs_list.append((zip_name, tb_text))

        all_my_eval_stats = all_my_eval_stats + curr_stats

    return all_my_eval_stats, all_my_excs_list

def extract_std(model, method):
    noise_rng_std = 1.
    if method in ('ppo1', 'ddpg_sb', 'td3', 'acktr', 'trpo'):
        std_ = -1.
        if method in ('ppo1', 'trpo'):
            for name, param_np in model.get_parameters().items():
                if 'logstd' in name:
                    std_ = np.sqrt(np.mean(np.exp(param_np)**2))
                    break
        elif method in ('ddpg_sb', 'td3'):
            std_ = model.action_noise._sigma

        if np.isnan(std_):
            print(f'Found a nan standard deviation {std_} in {ckpt_path}.')
        else:
            assert std_ >= 0., f'logstd was not found in parameters: {std_} in {ckpt_path}'
        return std_ * noise_rng_std
    elif method in ('ddpg',):
        return noise_rng_std
    elif method in ('sac',):
        return -1. #There is no meaning to it!
    elif method in ('wppo', 'wtrpo'):
        return -1. #Should be added manually

def get_test_a(obs, model, determinisim, method):
    if method in ('ppo1', 'ddpg_sb', 'td3', 'sac', 'acktr', 'trpo'):
        a, _ = model.predict(np.array([obs]), deterministic=determinisim)
        a = a[0]
    elif method in ('ddpg'):
        a = model.predict(np.array([obs]).reshape(1,-1), deterministic=determinisim).reshape(-1)
    elif method in ('wppo', 'wtrpo'):
        with torch.no_grad():
            a, _ = model( torch.from_numpy(np.array([obs])).double() )
            a = a[0].numpy()
    else:
        raise Exception(f'Not implemented yet')
    return a

def eval_ckpts_stepper(env, my_ckpt_params_dict, eval_ntrajs, eval_nsteps,
                       eval_seed, determinisim, model, method):
    all_my_eval_stats = []
    all_my_excs_list = []
    for zip_name, named_params in my_ckpt_params_dict.items():
        try:
            model.load_parameters(named_params)

            model.set_random_seed(eval_seed) # For controlling the policy randomizations
            np.random.seed(eval_seed) # For TD3 action noise seeding (sb noise classes use np.random)
            env.seed(eval_seed) # This makes sure all actors get the same initializations
            from stable_baselines.common.vec_env import DummyVecEnv
            is_dummy_env = isinstance(env, DummyVecEnv)
            curr_stats = []
            for traj_idx in range(eval_ntrajs):
                obs = env.reset()
                obs = obs[0] if is_dummy_env else obs
                traj_return = 0
                done = False
                NaNMujocoErr = False
                for traj_nsteps in range(10000):
                    a = get_test_a(obs, model, determinisim, method)
                    a = a.reshape(1, *a.shape) if is_dummy_env else a
                    from mujoco_py.builder import MujocoException
                    try:
                        obs, r, done, info = env.step(a)
                    except MujocoException as muj_err:
                        NaNMujocoErr = True
                        print(f'Mujoco exception {str(muj_err)} when generating evaluation trajectories.')
                        r, done= np.nan, np.nan
                    obs = obs[0] if is_dummy_env else obs
                    r = r[0] if is_dummy_env else obs
                    traj_return += r
                    if done:
                        break
                if not done:
                    raise Exception('The environment has not reported being done after 10000 steps')

                traj_seed = f'{eval_seed}_{traj_idx}'
                act_std = extract_std(model, method)
                stat_dict = dict(ckpt_name=zip_name, traj_return=traj_return,
                                 eval_seed=traj_seed, traj_nsteps=traj_nsteps+int(done),
                                 act_std=act_std)
                curr_stats.append(stat_dict)
        except Exception as eval_exc:
            curr_stats = []
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_text = traceback.format_exception(exc_type, exc_value,
                                                 exc_traceback,
                                                 limit=None, chain=True)
            all_my_excs_list.append((zip_name, tb_text))

        all_my_eval_stats = all_my_eval_stats + curr_stats

    return all_my_eval_stats, all_my_excs_list

if __name__ == '__main__':
    use_argparse = True
    if use_argparse:
        import argparse
        my_parser = argparse.ArgumentParser()
        my_parser.add_argument('-c', '--config_path', action='store', type=str, required=True)
        my_parser.add_argument('-s', '--nodes_size', action='store', type=int, required=True)
        my_parser.add_argument('-r', '--nodes_rank', action='store', type=int, required=True)
        my_parser.add_argument('-e', '--exec_id', action='store', type=str, required=True)
        args = my_parser.parse_args()
        args_config_path = args.config_path
        args_nodes_size = args.nodes_size
        args_nodes_rank = args.nodes_rank
        args_exec_id = args.exec_id
    else:
        args_config_path = './configs/myconf/myconf.json'
        args_nodes_size = 1
        args_nodes_rank = 0
        args_exec_id = -1

    ########################################
    ## Getting all experimental settings  ##
    ########################################
    cfg2hp_outdict = cfg2hp(args_config_path)
    all_settings = cfg2hp_outdict['all_settings']
    root_storage_dir = cfg2hp_outdict['root_storage_dir']
    train_set = cfg2hp_outdict['train_set']
    eval_set = cfg2hp_outdict['eval_set']
    method = cfg2hp_outdict['method']

    all_settings = [x for i, x in enumerate(all_settings)
                    if (i % args_nodes_size == args_nodes_rank)]

    runcountfile = os.environ.get('RUNCNTFILE', None)
    if runcountfile is not None:
        with open(runcountfile, 'w') as fp:
            fp.write(f'{2*len(all_settings)}')

    ########################################
    ########## Importing mpi4py  ###########
    ########################################
    # Since PPO2 doesn't need mpi4py, we should
    # be careful not to import it neither in the
    # main process nor the child subprocesses.
    mpi_tuple = initialize_mpi4py(method)
    mpi_comm, mpi_rank, mpi_size = mpi_tuple
    unnecessary_wait(mpi_rank)

    ########################################
    #### Attaching do_train and do_eval ####
    ########################################
    # The following adds a "do_train" and a
    # "do_eval" key to all dictionaries inside
    # the all_settings list.
    if mpi_rank == 0:
        os.makedirs(f'{root_storage_dir}/runbooks', exist_ok=True)
        runbook_path = f'{root_storage_dir}/runbooks/runbook_{args_nodes_rank}.csv'
        rb_manager = RunbookManager(runbook_path, args_exec_id)
        unnecessary_wait(mpi_rank)
        rb_manager.lock_and_load()
        rb_manager.add_missing_error_events()
        rb_manager.flush_and_unlock()

        for setting_dict in all_settings:
            config_id = setting_dict['config_id']
            train_done_path = setting_dict['train_done_path']
            eval_done_path = setting_dict['eval_done_path']
            conditions = dict(config_id=config_id)
            past_runs_df = rb_manager.get_cfgidx_finish_events(conditions)
            do_train, do_eval = should_train_eval(past_runs_df, train_set, eval_set,
                                                  train_done_path, eval_done_path,
                                                  args_exec_id)
            setting_dict['do_train'] = do_train
            setting_dict['do_eval'] = do_eval

    if mpi_comm is not None:
        all_settings = mpi_comm.bcast(all_settings, root=0)
    ########################################################
    not_done_list = list(x['do_train'] or x['do_eval'] for x in all_settings)

    if not any(not_done_list):
        if runcountfile is not None:
            with open(runcountfile, 'w') as fp:
                fp.write('0')
        sys.exit(0)

    next_setting_idx = int(np.argmax(not_done_list))

    setting_dict = all_settings[next_setting_idx]

    if mpi_rank == 0:
        unnecessary_wait(mpi_rank)
        rb_manager.lock_and_load()
        rb_manager.register_event(config_id=setting_dict['config_id'],
                                  event='start', do_train=setting_dict['do_train'],
                                  do_eval=setting_dict['do_eval'])
        rb_manager.flush_and_unlock()

    final_event = 'finish'
    try:
        unnecessary_wait(mpi_rank)
        finished_well = train_and_eval(setting_dict, mpi_tuple)
        if not finished_well:
            final_event = 'error'
    except Exception as exc:
        final_event = 'error'
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_text = traceback.format_exception(exc_type, exc_value,
                                             exc_traceback,
                                             limit=None, chain=True)
        tb_text = ''.join(tb_text)
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print('#' * 90)
        print('#' * 25 + f"  Catching a train_and_eval Exception   " + '#' * 25)
        print('#' * 25 + f" Date and Time <--> {dt_string} " + '#' * 25)
        print('#' * 90)
        print('The following setting caused the train_and_eval function to crash:')
        for key, val in setting_dict.items():
            print(f'  -> {key}: {val}')
        print('*'*20)
        print('Here is the exception that was caught:\n')
        print(tb_text)
        print('-' * 90 + '\n')

    if (mpi_rank == 0) and (final_event == 'finish'):
        rb_manager.lock_and_load()
        rb_manager.register_event(config_id=setting_dict['config_id'],
                                  event=final_event, do_train=setting_dict['do_train'],
                                  do_eval=setting_dict['do_eval'])
        rb_manager.flush_and_unlock()

    # if mpi_comm is not None:
    #     mpi_comm.Abort()
