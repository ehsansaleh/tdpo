from contextlib import contextmanager
import ctypes
import io
import os, sys, traceback
import fcntl
import tempfile
import json
import time
import shutil
from datetime import datetime
import tarfile
from git import Repo # gitpython
import socket
from os.path import abspath, exists, basename
from collections import OrderedDict as odict
from itertools import product

def unnecessary_wait(mpi_rank):
    time.sleep(mpi_rank * 0.1)

import numpy as np
import pandas as pd

libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')
c_stderr = ctypes.c_void_p.in_dll(libc, 'stderr')

STDOUTERR_MODE='w'
# https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
@contextmanager
def stdout_redirector(file_path, pre_finalizer=None, abortion=None):
    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()

    def _redirect_stdout_stderr(to_out_fd, to_err_fd=None):
        """Redirect stdout to the given file descriptor."""
        if to_err_fd is None:
            to_err_fd = to_out_fd
        # Flush the C-level buffer stdout
        libc.fflush(c_stdout)
        libc.fflush(c_stderr)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        sys.stderr.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_out_fd, original_stdout_fd)
        os.dup2(to_err_fd, original_stderr_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'), line_buffering=True)
        sys.stderr = io.TextIOWrapper(os.fdopen(original_stderr_fd, 'wb'), line_buffering=True)

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    saved_stderr_fd = os.dup(original_stderr_fd)

    perform_abortion = False
    try:
        tfile = open(file_path, STDOUTERR_MODE)
        _redirect_stdout_stderr(tfile.fileno())
        yield
        if pre_finalizer is not None:
            pre_finalizer()
    except Exception as exc:
        perform_abortion = True
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_text = traceback.format_exception(exc_type, exc_value,
                                             exc_traceback,
                                             limit=None, chain=True)
        tb_text = ''.join(tb_text)
        print(tb_text)
        if pre_finalizer is not None:
            pre_finalizer()
    finally:
        _redirect_stdout_stderr(saved_stdout_fd, saved_stderr_fd)
        tfile.flush()
        tfile.close()
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        if perform_abortion and (abortion is not None):
            abortion()

def cfg2hp(config_path):
    """
    This functions gets a json file config dictionary, and returns a list of
    dictionaries, each suited for running on a single node at one time without
    any loops.
    """
    config_path = abspath(config_path)
    assert '/configs/' in config_path, '''the configs directory marks the project
                                          root, and must exist in the config path'''
    assert config_path.endswith('.json'), f'Not a .json file: {config_path}'

    config_path_split = tuple(config_path.split('/configs/'))
    msg_ = f'More than one "/configs/" was found in your config_path: {config_path}'
    assert len(config_path_split) == 2, msg_
    root_proj_dir, config_tree_and_name = config_path_split
    if '/' in config_tree_and_name:
        configid_split = config_tree_and_name.split('/')
        config_name = configid_split[-1][:-5] # [:-5] is to remove the .json extension
        config_tree = '/'.join(configid_split[:-1])
    else:
        config_id = config_tree_and_name[:-5] # [:-5] is to remove the .json extension
        config_tree = ''

    cfg_path = abspath(f'{root_proj_dir}/configs/{config_tree}/{config_name}.json')
    print(f'Reading Configuration from {cfg_path}', flush=True)
    with open(cfg_path) as f:
        config_dict = json.load(f, object_pairs_hook=odict)

    method = config_dict['method']
    assert method in ('trpo', 'ppo2', 'ppo1', 'td3', 'wtrpo') #This is what's supported so far

    logistic_keys = ['results_dir_name', 'storage_dir_name',
                     'train_set', 'eval_set', 'OPENAI_LOG_FORMAT',
                     'walltime_hrs', 'num_checkpoints']
    opt_keys = ['description']
    base_keys = ['method', 'environment', 'looping_tree', 'num_envs']

    trpo_init_keys = ['timesteps_per_batch', 'gamma', 'lam', 'max_kl',
                      'policy_kwargs', 'cg_iters', 'entcoeff', 'cg_damping',
                      'vf_stepsize', 'vf_iters', 'vf_minibatches']

    ppo2_init_keys = ['gamma', 'n_steps', 'ent_coef', 'learning_rate',
                      'policy_kwargs', 'vf_coef', 'max_grad_norm',
                      'lam', 'nminibatches', 'noptepochs', 'cliprange']

    ppo1_init_keys = ['gamma', 'timesteps_per_actorbatch', 'clip_param',
                      'policy_kwargs', 'entcoeff','optim_epochs',
                      'optim_stepsize', 'optim_minibatches', 'lam',
                      'adam_epsilon', 'schedule']

    td3_init_keys = ['gamma', 'buffer_size', 'learning_starts',
                     'train_freq', 'policy_kwargs', 'batch_size',
                     'learning_rate', 'gradient_steps', 'tau',
                     'policy_delay', 'actnoise_type', 'actnoise_freqrel',
                     'actnoise_stdrel', 'target_policy_noise',
                     'target_noise_clip', 'random_exploration']

    wtrpo_init_keys = ['gamma', 'c_w2', 'c_wg', 'n_vine', 'policy_kwargs',
                       'explrn_init', 'explrn_lr', 'explrn_steps',
                       'cg_damping', 'cg_iters', 'cg_eps',
                       'ls_num', 'ls_span', 'ls_reps']

    eval_keys = ["eval_ntrajs", "eval_nsteps", "eval_seed"]

    if method == 'trpo':
        method_init_keys = trpo_init_keys
    elif method == 'ppo2':
        method_init_keys = ppo2_init_keys
    elif method == 'ppo1':
        method_init_keys = ppo1_init_keys
    elif method == 'td3':
        method_init_keys = td3_init_keys
    elif method == 'wtrpo':
        method_init_keys = wtrpo_init_keys
    else:
        raise Exception(f'Not implemented for method {method}')

    method_etc_keys = ['total_timesteps', 'rng_seed']
    method_keys = method_init_keys + method_etc_keys

    ################################################################################
    ############################## Importing MPI4PY ################################
    ################################################################################
    if method in ('trpo', 'ppo1', 'td3', 'wtrpo'):
        from mpi4py import MPI
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        mpi_size = mpi_comm.size
    elif method in ('ppo2',):
        mpi_comm = None
        mpi_rank = 0
        mpi_size = 1
    else:
        raise ValueError(f'method {method} not implemented.')

    ################################################################################
    ############################## Config Assertions ###############################
    ################################################################################
    for key in base_keys:
        assert key in config_dict, f'{key} must be specified in the config file.'

    allowed_keys = logistic_keys + opt_keys + base_keys + method_keys + eval_keys

    non_specified_keys = set(allowed_keys).difference(set(config_dict.keys()))
    over_specified_keys = set(config_dict.keys()).difference(set(allowed_keys))

    msg_ = f'You have to specify these keys in the config file: {non_specified_keys}'
    assert len(non_specified_keys) == 0, msg_
    msg_ = f'I am not sure what to do with these keys: {over_specified_keys}'
    assert len(over_specified_keys) == 0, msg_

    train_set = config_dict['train_set']
    eval_set = config_dict['eval_set']

    ################################################################################
    ############################## Res/Storage Roots ###############################
    ################################################################################
    results_dir_name = config_dict['results_dir_name']
    storage_dir_name = config_dict['storage_dir_name']

     # The csv/summary files
    root_results_dir = f'{root_proj_dir}/{results_dir_name}/{config_tree}/{config_name}'
    # The heavy stuff
    root_storage_dir = f'{root_proj_dir}/{storage_dir_name}/{config_tree}/{config_name}'

    # Making a trash directory to put old finished runs inside it
    now = datetime.now()
    today_str = now.strftime("%d_%m_%Y")
    trash_idx = 0
    while True:
        root_trash_dir = f'{root_proj_dir}/trash/{today_str}_{trash_idx}' # The old runs
        if not exists(root_trash_dir):
            break
        trash_idx += 1

    if mpi_comm is not None:
        bcast_dict = dict(root_trash_dir=root_trash_dir) if mpi_rank == 0 else None
        root_trash_dir = mpi_comm.bcast(bcast_dict, root=0)['root_trash_dir']
    root_trash_results_dir = f'{root_trash_dir}/{results_dir_name}/{config_tree}/{config_name}'
    root_trash_storage_dir = f'{root_trash_dir}/{storage_dir_name}/{config_tree}/{config_name}'

    ################################################################################
    ########################### Processing looping_tree ############################
    ################################################################################
    looping_tree = config_dict['looping_tree']
    looping_vars = method_keys + ['method', 'environment']
    non_specified_keys = set(looping_vars).difference(set(looping_tree.keys()))
    over_specified_keys = set(looping_tree.keys()).difference(set(looping_vars))

    msg_ = f'You have to specify these keys in the looping_tree: {non_specified_keys}'
    assert len(non_specified_keys) == 0, msg_
    msg_ = f'Extra keys in the looping_tree: {over_specified_keys}'
    assert len(over_specified_keys) == 0, msg_
    assert looping_tree['method'] == 'fixed', 'not supported'

    looping_keys = list(looping_tree.keys())
    looping_vals = list(looping_tree.values())

    ########################################
    ########### Checking Orders ############
    ########################################
    # 1) Making sure the looping tree and the config dict have the same variable order
    cfg_key_ord = [key for key in config_dict.keys() if key in set(looping_keys)]
    msg_  = f'The config dictionary and the looping tree do no have the same variables order:\n'
    msg_ += f' --> config  dict order: {looping_keys}'
    msg_ += f' --> looping tree order: {cfg_key_ord}'
    assert looping_keys == cfg_key_ord, msg_

    # 2) Making sure ovat keys come after cartesian keys
    if 'ovat' in looping_vals:
        first_ovat_idx = looping_vals.index('ovat')
    else:
        first_ovat_idx = len(looping_vals)

    if 'cartesian' in looping_vals:
        last_cartesian_idx = max(i for i, val in enumerate(looping_vals) if val == 'cartesian')
    else:
        last_cartesian_idx = -1

    msg_  = f'ovat loopings cannot occur before cartesian searches.\n'
    msg_ += f'first_ovat_idx = {first_ovat_idx}\n'
    msg_ += f'last_cartesian_idx = {last_cartesian_idx}\n'
    assert first_ovat_idx > last_cartesian_idx, msg_

    # 3) Making sure ovat and cartesian configs are lists
    ovat_keys = [key for key, val in looping_tree.items() if val == 'ovat']
    zip_keys = [key for key, val in looping_tree.items() if val == 'zip']
    fixed_keys = [key for key, val in looping_tree.items() if val == 'fixed']
    cartesian_keys = [key for key, val in looping_tree.items() if val == 'cartesian']
    cartesian_candids = [config_dict[key] for key in cartesian_keys]
    zip_candids = [config_dict[key] for key in zip_keys]

    for key in ovat_keys:
        msg_ = f'The key {key} has the ovat looping method, but its config value is not a list.'
        assert isinstance(config_dict[key], list), msg_
    for key in zip_keys:
        msg_ = f'The key {key} has the zip looping method, but its config value is not a list.'
        assert isinstance(config_dict[key], list), msg_
    for key in cartesian_keys:
        msg_ = f'The key {key} has the cartesian looping method, but its config value is not a list.'
        assert isinstance(config_dict[key], list), msg_

    zipcount = 0
    if len(zip_keys) > 0:
        zipcount = len(zip_candids[0])
        assert zipcount > 0, 'The zip candidate lists must not be empty.'
        msg_ = f'all zip lengths must be the same'
        assert all(len(cnd) == zipcount for cnd in zip_candids), msg_

    ################################################################################
    ######################## Generating Individual Configs #########################
    ################################################################################
    pass_along_keys = ['OPENAI_LOG_FORMAT', 'walltime_hrs', 'num_checkpoints', 'num_envs']
    pass_along_keys = pass_along_keys + eval_keys

    main_cfg = dict()
    # main_cfg will hold two kinds of keys:
    #   1) The ovat keys with their first candidates
    #   2) The fixed keys
    #   3) The pass-along keys
    for key in ovat_keys:
        main_cfg[key] = config_dict[key][0]
    for key in fixed_keys:
        main_cfg[key] = config_dict[key]
    for key in pass_along_keys:
        main_cfg[key] = config_dict[key]

    ########################################
    ####### Generating Ovat Configs ########
    ########################################
    ovat_fixed_cfgs = [main_cfg]
    for key in ovat_keys:
        for cfg_val in config_dict[key][1:]:
            d = main_cfg.copy()
            d[key] = cfg_val
            ovat_fixed_cfgs.append(d)

    ########################################
    ##### Generating Cartesian Configs #####
    ########################################
    cartesian_cfgs = [dict(zip(cartesian_keys, instance))
                      for instance in product(*cartesian_candids)]

    ########################################
    ######## Generating Zip Configs ########
    ########################################
    zip_cfgs = [{key: config_dict[key][i] for key in zip_keys}
                for i in range(zipcount)]
    zip_cfgs = zip_cfgs or [dict()]

    ########################################
    ######### Merging All Configs ##########
    ########################################
    all_cfgs = []
    cfg_idx = 0
    for cart_cfg in cartesian_cfgs:
        for ovat_fixed_cfg in ovat_fixed_cfgs:
            for zip_cfg in zip_cfgs:
                d = cart_cfg.copy()
                d.update(ovat_fixed_cfg)
                d.update(zip_cfg)
                config_id = f'cfg_{cfg_idx}'
                results_dir = f'{root_results_dir}/{config_id}'
                storage_dir = f'{root_storage_dir}/{config_id}'
                train_done_path = f'{storage_dir}/train.done'
                eval_done_path = f'{storage_dir}/eval.done'

                d['config_id'] = config_id
                d['results_dir'] = results_dir
                d['storage_dir'] = storage_dir
                d['trash_results_dir'] = f'{root_trash_results_dir}/{config_id}'
                d['trash_storage_dir'] = f'{root_trash_storage_dir}/{config_id}'
                d['train_done_path'] = train_done_path
                d['eval_done_path'] = eval_done_path

                all_cfgs.append(d)
                cfg_idx += 1

    ########################################
    ############# Outputting ###############
    ########################################
    out_dict = dict(all_settings=all_cfgs,
                    root_storage_dir=root_storage_dir,
                    train_set=train_set,
                    eval_set=eval_set,
                    method=method)

    return out_dict

def oldify_and_rename_file_if_exists(file_path):
    if exists(file_path):
        oldidx = 0
        while True:
            log_path_old = f'{file_path}_old{oldidx}'
            if exists(log_path_old):
                oldidx += 1
            else:
                shutil.move(file_path, log_path_old)
                break

class RunbookManager:
    x_columns = ('config_id', 'hostname', 'exec_id', 'commit_hash')
    y_columns = ('event', 'do_train', 'do_eval', 'date', 'timestamp',
                 'is_commit_uptodate')
    columns = tuple(list(x_columns) + list(y_columns))

    def __init__(self, runbook_path, exec_id):
        self.runbook_path = runbook_path
        self.exec_id = exec_id

        git_info = get_git_info()
        self.commit_hash = git_info['commit_hash']
        self.is_commit_uptodate = git_info['is_uptodate']

        self.date, self.timestamp = self.get_date_timestamp()
        self.hostname = socket.gethostname()
        self.df = None
        self.is_locked = False
        self.runbook_fileobj = None

    def lock_and_load(self):
        self.runbook_fileobj = open(self.runbook_path, 'a+')
        fcntl.flock(self.runbook_fileobj.fileno(), fcntl.LOCK_EX)

        self.runbook_fileobj.seek(0, 2)
        fobj_size = self.runbook_fileobj.tell()
        self.runbook_fileobj.seek(0)

        if fobj_size > 0:
            self.df = pd.read_csv(self.runbook_fileobj)
        else: # not exists(self.runbook_path)
            self.df = pd.DataFrame(columns=self.columns)

        assert set(self.df.columns) == set(self.columns), f'{df.columns} != {self.columns}'

        self.assert_df_is_valid(self.df)
        self.is_locked = True

    def flush_and_unlock(self):
        assert self.is_locked
        self.assert_df_is_valid(self.df)

        self.runbook_fileobj.truncate(0) # Wiping out the existing contents
        self.runbook_fileobj.seek(0)
        self.df.to_csv(self.runbook_fileobj, index=False)
        self.runbook_fileobj.close()

        self.is_locked = False

    def get_date_timestamp(self):
        now = datetime.now()
        date = now.strftime("%d/%m/%Y %H:%M:%S")
        timestamp = now.timestamp()
        return date, timestamp

    def assert_df_is_valid(self, df):
        assert set(df['event'].tolist()).issubset({'start', 'finish', 'error'})

    def add_missing_error_events(self):
        self.assert_df_is_valid(self.df)
        df = self.df

        myexec_df =  df[df['exec_id'] == self.exec_id]
        error_rows_lst = []
        for x_cfg, x_df in myexec_df.groupby(list(self.x_columns)):
            n_rows = x_df.shape[0]
            if n_rows == 0:
                continue
            x_df_ = x_df.sort_index(inplace=False)
            buttom_row = x_df_.iloc[-1]
            if buttom_row['event'] == 'start':
                assert self.hostname == buttom_row['hostname']
                assert self.exec_id == buttom_row['exec_id']
                assert self.commit_hash == buttom_row['commit_hash']
                error_dict = dict()
                for key in self.columns:
                    error_dict[key] = buttom_row[key]
                error_dict['event'] = 'error'
                error_dict['date'], error_dict['timestamp'] = self.get_date_timestamp()
                error_rows_lst.append(error_dict)
            elif buttom_row['event'] in ('finish', 'error'):
                pass
            else:
                raise RuntimeError(f"unknown event {buttom_row['event']}.")
        error_df = pd.DataFrame(error_rows_lst)
        ammended_df = pd.concat([df, error_df], ignore_index=True).copy(deep=True)

        self.df = ammended_df
        self.assert_df_is_valid(self.df)

    def get_cfgidx_finish_events(self, conditions):
        df = self.df

        assert isinstance(conditions, dict)
        assert 'config_id' in conditions
        assert set(conditions.keys()).issubset(self.columns)

        conditions = conditions.copy()
        for key, val in conditions.items():
            df = df[df[key] == val]

        df = df[(df['event'] == 'finish') | (df['event'] == 'error')]
        return df

    def register_event(self, config_id, event, do_train, do_eval):
        self.assert_df_is_valid(self.df)
        df = self.df

        date, timestamp = self.get_date_timestamp()
        row_dict = dict(config_id=config_id, hostname=self.hostname,
                        exec_id=self.exec_id, commit_hash=self.commit_hash,
                        event=event, do_train=do_train, do_eval=do_eval,
                        date=date, timestamp=timestamp,
                        is_commit_uptodate=self.is_commit_uptodate)
        assert self.columns

        row_df = pd.DataFrame(row_dict, index=[0])
        ammended_df = pd.concat([df, row_df], ignore_index=True).copy(deep=True)
        self.df = ammended_df
        self.assert_df_is_valid(self.df)


def get_git_info():
    repo = Repo('./', search_parent_directories=True)
    diffs_list = repo.index.diff(None)
    commit_hash = repo.head.object.hexsha
    is_uptodate = len(diffs_list) > 0
    out_dict = dict(commit_hash=commit_hash,
                    is_uptodate=is_uptodate,
                    diffs_list=diffs_list)
    return out_dict


class SBCallback:
    def __init__(self, method, walltime_hrs, wall_samples, num_checkpoints,
                 tar_path, mpi_rank, mpi_size, mpi_comm):
        self.method = method
        self.walltime_hrs = walltime_hrs
        self.wall_samples = wall_samples
        self.num_checkpoints = num_checkpoints
        self.tar_path = tar_path
        self.last_ckpt_step = -1e10
        self.init_loop_time = None
        self.last_ckpt_time = None
        self.mpi_rank = mpi_rank
        self.mpi_size = mpi_size
        self.mpi_comm = mpi_comm

        if self.method in ('ppo1', 'trpo'):
            self.steps_var = 'timesteps_so_far'
            self.steps_coeff = 1
        elif self.method == 'DDPG':
            self.steps_var = 'total_steps'
            self.steps_coeff = self.mpi_size
        elif self.method == 'td3':
            self.steps_var = 'sofar_timesteps_allranks'
            self.steps_coeff = 1
        elif self.method == 'td3':
            self.steps_var = 'num_timesteps'
            self.steps_coeff = self.mpi_size
        elif self.method == 'ppo2':
            self.steps_var = None
            self.steps_coeff = 1
        else:
            raise ValueError(f'Unknown alg {alg}')
        self.archive = None
        self.archive_fp = None
        self.waltime_secs = self.walltime_hrs * 3600.
        self.walltime_div_ckpts = self.waltime_secs / self.num_checkpoints
        self.wallsamp_div_ckpts = float(self.wall_samples) / self.num_checkpoints
        self.reached_infinite_params = False
        self.reached_walltime = False
        self.total_timesteps = 0
        self.total_callbacks = 0

    def append_to_tar(self, file_name, file_like_obj):
        if self.archive is None:
            # self.archive = tarfile.open(self.tar_path, "w")
            # self.archive_fp = io.BytesIO()
            self.archive_fp = open(self.tar_path, 'wb', buffering=0)
            self.archive = tarfile.open(fileobj=self.archive_fp, mode="w")
        info = tarfile.TarInfo(name=file_name)
        file_like_obj.seek(0, io.SEEK_END)
        info.size = file_like_obj.tell()
        file_like_obj.seek(0, io.SEEK_SET)
        info.mtime = time.time()
        self.archive.addfile(info, file_like_obj)
        file_like_obj.close()

    def is_mdl_finite(self, mdl):
        params_dict = mdl.get_parameters()
        is_finite = all(np.isfinite(param).all() for name, param in params_dict.items())
        return is_finite

    def __call__(self, locals_dict, globals_dict):
        mdl = locals_dict['self']
        if self.method == 'ppo2':
            num_steps = mdl.num_timesteps
        elif (self.method == 'td3') and ('sofar_timesteps_allranks' not in locals_dict):
            num_steps = mdl.num_timesteps * self.mpi_size
        else:
            num_steps = locals_dict[self.steps_var] * self.steps_coeff

        self.total_timesteps = num_steps
        self.total_callbacks += 1

        if self.method == 'ddpg':
            # These lists never flush and only increase in size, causing
            # memory allocation errors later in averaging lines. Due to
            # the memory allocation errors, we have to limit their size.
            mem_leaks = ['epoch_actions', 'epoch_qs', 'epoch_episode_rewards',
                         'episode_rewards_history', 'epoch_episode_steps']
            for unflushed_list_name in mem_leaks:
                 if len(loc_[unflushed_list_name]) > 10000:
                     del loc_[unflushed_list_name][0]

        done_ = False
        if self.mpi_rank == 0:
            time_now_ = time.time()
            if self.init_loop_time is None:
                self.init_loop_time = time_now_
                self.last_ckpt_time = time_now_

            step_freq_check = (num_steps - self.last_ckpt_step) >= self.wallsamp_div_ckpts
            time_freq_check = (time_now_ - self.last_ckpt_time) >= self.walltime_div_ckpts
            if step_freq_check or time_freq_check:
                zip_filename = f'ckpt_{num_steps}.zip'
                file_like_obj = io.BytesIO()
                mdl.save(file_like_obj, cloudpickle=False)
                file_like_obj.seek(0)
                self.append_to_tar(zip_filename, file_like_obj)
                self.last_ckpt_step = num_steps
                self.last_ckpt_time = time_now_

            self.reached_infinite_params = not(self.is_mdl_finite(mdl))
            self.reached_walltime = (time_now_ - self.init_loop_time) > self.waltime_secs
            done_ = self.reached_infinite_params or self.reached_walltime

            if done_:
                self.close()

        if self.mpi_comm is not None:
            done_ = self.mpi_comm.bcast(done_, root=0)

        return not(done_)

    def get_completion_info(self):
        d = dict(reached_infinite_params = self.reached_infinite_params,
                 reached_walltime = self.reached_walltime,
                 total_timesteps = self.total_timesteps,
                 total_callbacks = self.total_callbacks)
        return d

    def close(self):
        if (self.mpi_rank == 0) and (self.archive is not None):
            self.archive.close()
            self.archive_fp.close()

def should_train_eval(past_runs_df, train_set, eval_set,
                      train_done_path, eval_done_path, my_execid):
    """
    Determines do_train and do_eval based on the past runs from the runbook,
    and the contents of train.done

    :param: past_runs_df(df): a pandas data-frame coming from the RunbookManager
                              (see the x_columns and y_columns of the RunbookManager).
    :param: train_set(str): one of ['remaining', 'errors', 'none']
    :param: eval_set(str): one of ['remaining', 'none']
    :param: train_done_path(str): the path to the train.done file
    :param: eval_done_path(str): the path to the eval.done file
    :param: my_execid(str): the execution id of this run (used for not runing an errory
                            run twice in the same execution)
    """

    assert train_set in ('remaining', 'errors', 'custom1', 'none')
    assert eval_set in ('remaining', 'errors', 'train_set', 'none')

    past_trains_df = past_runs_df[past_runs_df['do_train'] == True]
    past_evals_df1 = past_runs_df[(past_runs_df['do_eval'] == True) &
                                  (past_runs_df['do_train'] == True)]
    past_evals_df2 = past_runs_df[(past_runs_df['do_eval'] == True) &
                                  (past_runs_df['do_train'] == False)]

    past_train_events = past_trains_df['event'].tolist()
    past_eval_events1 = past_evals_df1['event'].tolist()
    past_eval_events2 = past_evals_df2['event'].tolist()

    is_train_done_filled = exists(train_done_path)
    has_train_finished = ('finish' in past_train_events)
    has_train_errored = ('error' in past_train_events)

    is_eval_done_filled  = exists(eval_done_path)
    has_eval_finished = (('finish' in past_eval_events1) or
                         ('finish' in past_eval_events2))
    has_eval_errored = ('error' in past_eval_events2)

    # Repeating the above only for the error cases and preventing an
    # infinite loop if we error again on past errored runs
    myexec_runs_df = past_runs_df[past_runs_df['exec_id'] == my_execid]
    myexec_trains_df = myexec_runs_df[myexec_runs_df['do_train'] == True]
    myexec_evals_df2 = myexec_runs_df[(myexec_runs_df['do_eval'] == True) &
                                      (myexec_runs_df['do_train'] == False)]
    myexec_train_events = myexec_trains_df['event'].tolist()
    myexec_eval_events2 = myexec_evals_df2['event'].tolist()
    has_myexec_train_finished = ('finish' in myexec_train_events)
    has_myexec_train_errored = ('error' in myexec_train_events)
    has_myexec_eval_errored = ('error' in myexec_eval_events2)

    if train_set == 'remaining':
        do_train = not(has_train_finished or has_train_errored)
    elif train_set == 'errors':
        do_train = has_train_errored and not(has_train_finished) and not(has_myexec_train_errored)
    elif train_set == 'none':
        do_train = False
    elif train_set == 'custom1':
        a = [10, 109, 11, 110, 111, 183, 184, 185, 196, 197,
             198, 22, 23, 24, 9, 96, 97, 98, 221]
        b = [f'/2_ovat5b_eng/2_ppo2/cfg_{x}/' for x in a]
        do_train = any(x in train_done_path for x in b)
        do_train = do_train and not(has_myexec_train_errored or has_myexec_train_finished)
    else:
        raise RuntimeError(f'train_set {train_set} rule not implemented.')

    if eval_set == 'remaining':
        do_eval = not(has_eval_finished or has_eval_errored)
    elif eval_set == 'errors':
        do_eval = has_eval_errored and not(has_eval_finished) and not(has_myexec_eval_errored)
    elif eval_set == 'train_set':
        do_eval = do_train
    elif eval_set == 'none':
        do_eval = False
    else:
        raise RuntimeError(f'eval_set {eval_set} rule not implemented.')

    return do_train, do_eval

def initialize_mpi4py(method):
    if method in ('trpo', 'ppo1', 'td3', 'wtrpo'):
        from mpi4py import MPI
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        mpi_size = mpi_comm.size
    elif method in ('ppo2',):
        # We should definitely not be running in MPI mode,
        # or the linux kernel could break!
        mpi_comm = None
        mpi_rank = 0
        mpi_size = 1
        for mpi_envvar in ['OMPI_COMM_WORLD_SIZE',
                           'OMPI_COMM_WORLD_RANK',
                           'OMPI_UNIVERSE_SIZE']:
            if mpi_envvar in os.environ:
                raise RuntimeError(f'The {method} variant should not run '\
                                   'with mpirun/mpiexec!')
        os.environ['NOMPI4PY'] = '1'
        # This will be used in `sb_utils.py` to disable mpi4py.
        # Environmental Variables are the only way of communicating to
        # spawned processes that they should disable mpi4py before
        # importing anything from stable_baselines.
    else:
        raise RuntimeError(f'method {method} not implemented.')
    return mpi_comm, mpi_rank, mpi_size

def scatter_ckpt_params_mpi(ckpt_params_dict, mpi_comm, mpi_rank, mpi_size):
    """
    This function gets a dictionary map of zip file names to named parameter dictionaries,
    and scatters them as evenly as possible among MPI workers.

    Parameters:
      * ckpt_params_dict (dict or None):
           This should be None for non-root members, and a dictionary for root. For the root,
           the keys are zip file names, and values are named parameter dictionaries.
           Example:
             ckpt_params_dict = {'ckpt_0.zip': dict(fc1 = fc1_w.T, fc2 = fc2_w.T,
                                                    fc3 = fc3_w.T, fc1_bias = fc1_b,
                                                    fc2_bias = fc2_b, fc3_bias = fc3_b,
                                                    policy_logstd = policy_logstd)}

    Returns:
      * my_ckpt_params_dict (dict):
          A dictionary with the same structure as input, but scattered among different
          MPI workers.

    """
    if mpi_rank == 0:
        zip_names = list(ckpt_params_dict.keys())
        named_params_list = list(ckpt_params_dict.values())
        num_ckpts = len(zip_names)
        scatter_size = int(np.ceil(num_ckpts / mpi_size)) * mpi_size
    else:
        scatter_size = None
        # zip_names, named_params_list, num_ckpts are left intentionally undefined,
        # since they shouldn't be accessed on non-root workers

    if mpi_rank == 0:
        example_params = named_params_list[0]
        actor_param_names = tuple(example_params.keys())
        actor_param_shapes = tuple(example_params[name].shape for name in actor_param_names)
        actor_param_dtypes = tuple(np.dtype(example_params[name].dtype).name for name in actor_param_names)
        # Ex. -->    np.dtype(np.int32).name == 'int32'
    else:
        actor_param_names = None
        actor_param_shapes = None
        actor_param_dtypes = None

    scatter_size = mpi_comm.bcast(scatter_size, root=0)
    actor_param_names = mpi_comm.bcast(actor_param_names, root=0)
    actor_param_shapes = mpi_comm.bcast(actor_param_shapes, root=0)
    actor_param_dtypes = mpi_comm.bcast(actor_param_dtypes, root=0)

    if mpi_rank == 0:
        scatter_zip_names = [None] * scatter_size
        for i, p_dict in enumerate(named_params_list):
            scatter_zip_names[i] = zip_names[i]
        scatter_params_dict = dict()
        for p_name, p_shape, p_dtype in zip(actor_param_names, actor_param_shapes, actor_param_dtypes):
            cat_param_arr = np.zeros((scatter_size, *p_shape), dtype=p_dtype)
            for i, p_dict in enumerate(named_params_list):
                cat_param_arr[i] = p_dict[p_name]
            scatter_params_dict[p_name] = cat_param_arr
        scatter_zip_names = np.array(scatter_zip_names).reshape(mpi_size,-1).tolist()
    else:
        scatter_zip_names = None
        scatter_params_dict = {p_name: None for p_name in actor_param_names}

    my_zip_names_w_nones = mpi_comm.scatter(scatter_zip_names, root=0)

    my_stacked_params_dict = dict()
    for _, (p_name, p_shape, p_dtype) in enumerate(zip(actor_param_names, actor_param_shapes, actor_param_dtypes)):
        my_stacked_params_dict[p_name] = np.empty((scatter_size//mpi_size, *p_shape), dtype=p_dtype)
        sendbuf = scatter_params_dict[p_name]
        recvbuf = my_stacked_params_dict[p_name]
        mpi_comm.Scatter(sendbuf, recvbuf, root=0)

    my_ckpt_params_dict = odict()
    for i, zip_name in enumerate(my_zip_names_w_nones):
        if zip_name is None:
            continue
        p_dict = {p_name: my_stacked_params_dict[p_name][i] for p_name in actor_param_names}
        my_ckpt_params_dict[zip_name] = p_dict
    return my_ckpt_params_dict

def scatter_dict_mpi_pkl(ckpt_params_dict, mpi_comm, mpi_rank, mpi_size):
    """
    This function gets a dictionary map of zip file names to named parameter dictionaries,
    and scatters them as evenly as possible among MPI workers.

    Note: This function should have the same functionality as scatter_ckpt_params_mpi, and
          should retrun identical outputs ideally. However, this function doesn't scatter
          buffered objects, and uses mpi4py's pickling protocols to scatter arbitrary objects.

    Parameters:
      * ckpt_params_dict (dict or None):
           This should be None for non-root members, and a pickle-able dictionary for root.

    Returns:
      * my_ckpt_params_dict (dict):
          A dictionary with the same structure as input, but scattered among different
          MPI workers.

    """

    if mpi_rank == 0:
        workers_ckpt_params_dicts = [dict() for _ in range(mpi_size)]
        for i, (key, val) in enumerate(ckpt_params_dict.items()):
            workers_ckpt_params_dicts[i % mpi_size][key] = val
    else:
        workers_ckpt_params_dicts = None
    my_ckpt_params_dict = mpi_comm.scatter(workers_ckpt_params_dicts, root=0)
    return my_ckpt_params_dict

def gather_stats_mpi(eval_stats, mpi_comm, mpi_rank, mpi_size):
    eval_stats_all = mpi_comm.gather(eval_stats, root=0)
    if mpi_rank == 0:
        agg_eval_stats = []
        for ev_st in eval_stats_all:
            agg_eval_stats = agg_eval_stats + ev_st
    else:
        agg_eval_stats = []

    return agg_eval_stats

def process_policy_kwargs(policy_kwargs, method):
    assert isinstance(policy_kwargs, dict) or (policy_kwargs is None)

    msg_kw = """
    The only acceptable policy_kwargs are:
    1) net_arch (list): only when method is either 'ppo1', 'trpo', or 'ppo2'
    2) layers (list)
    3) h1 (int): number of units in first hidden layer
    4) h2 (int): number of units in second hidden layer
    5) activation (str): either 'tanh' or 'relu'
    6) do_mlp_output_tanh (bool): should only be True for td3.
    7) mlp_output_scaling (float): should only be non-unit for td3.

    * Instead of act_fun, use the activation keyword argument from above.
    """

    if isinstance(policy_kwargs, dict):
        assert 'act_fun' not in policy_kwargs, msg_kw
        if (method in ('ppo1', 'trpo', 'ppo2', 'wtrpo')):
            assert not policy_kwargs.get('do_mlp_output_tanh', False), msg_kw
            assert policy_kwargs.get('mlp_output_scaling', 1) == 1, msg_kw
        elif method in ('td3',):
            msg_ =  'td3/ddpg policies dont support net_arch and they are '
            msg_ += 'programmed to ignore extra kwargs. Use layers instead.'
            assert 'net_arch' not in policy_kwargs, msg_
        else:
            raise ValueError(f'method {method} not implemented.')

    # determining h1, h2
    if policy_kwargs is None:
        h1, h2 = 64, 64
    elif 'net_arch' in policy_kwargs:
        assert method in ('ppo1', 'trpo', 'ppo2', 'wtrpo')
        assert isinstance(policy_kwargs['net_arch'], list)
        msg_ = 'such rolling policy is not implemented.'
        assert len(policy_kwargs['net_arch']) == 1, msg_
        assert set(policy_kwargs['net_arch'][0].keys()) == {'pi', 'vf'}, msg_
        assert len(policy_kwargs['net_arch'][0]['pi']) == 2, msg_
        h1, h2 = policy_kwargs['net_arch'][0]['pi']
    elif 'layers' in policy_kwargs:
        h1, h2 = policy_kwargs['layers']
    else:
        h1, h2 = 64, 64

    # dividing policy_kwargs into a stable_baselines portion, and an env portion
    if policy_kwargs is None:
        policy_kwargs_cpdict = dict()
    elif isinstance(policy_kwargs, dict):
        policy_kwargs_cpdict = policy_kwargs.copy() # shallow copy
    else:
        raise RunTimeError('policy_kwargs type not implemented')

    activation = policy_kwargs_cpdict.pop('activation', 'tanh')
    do_mlp_output_tanh = policy_kwargs_cpdict.pop('do_mlp_output_tanh', False)
    mlp_output_scaling = policy_kwargs_cpdict.pop('mlp_output_scaling', 1)

    assert activation in ('tanh', 'relu')
    assert do_mlp_output_tanh in (True, False)

    if 'activation' in policy_kwargs:
        policy_kwargs_cpdict['act_fun'] = activation

    policy_kwargs_sb = None if policy_kwargs is None else policy_kwargs_cpdict
    policy_kwargs_env = dict(activation=activation,
                             do_mlp_output_tanh=do_mlp_output_tanh,
                             mlp_output_scaling=mlp_output_scaling,
                             h1=h1, h2=h2)

    return policy_kwargs_sb, policy_kwargs_env
