import json
import sys
import os
from os.path import abspath, dirname, exists
from collections import OrderedDict as odict
import pandas as pd
import numpy as np
import random
import copy

PROJPATH = abspath(os.path.join(dirname(__file__), os.pardir))
#sys.path.insert(-1, f'{PROJPATH}/usr')

from cfg import smry_tbls_dir, smry_fmt
from cfg  import xcols, tcols, ycols, evalcols, ignore_cols
from csv2summ import main as csv2summ_main

self_logcasting = True # Whether to apply the log/casting myslef, or outsource it to the library
use_mpi = True

query_curriter_only = False
only_dryrun = False

ASSERT_UNQ_LOC = True

if use_mpi:
    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
else:
    mpi_rank, mpi_size = 0, 1

def ensure_pythonhashseed(seed=0, failure_action='restart_python'):
    import os
    current_seed = os.environ.get("PYTHONHASHSEED", None)
    seed = str(seed)
    if current_seed is None or current_seed != seed:
        if failure_action == 'restart_python':
            print(f'Resetting Python with PYTHONHASHSEED="{seed}"')
            os.environ["PYTHONHASHSEED"] = seed
            # restart the current process
            os.execl(sys.executable, sys.executable, *sys.argv)
        elif failure_action == 'raise_error':
            raise RuntimeError(f'Please restart me by setting \
                                 PYTHONHASHSEED="{seed}" for \
                                 reproducibility!')
        else:
            raise RunTimeError(f'Unknown action {failure_action}')

def amend_paramprops(hpo_paramprops):
    hpo_paramprops = copy.deepcopy(hpo_paramprops)

    for hpname, hpprops in hpo_paramprops.items():
        hpprops_cp = dict(hpprops) # shallow copy
        hptype = hpprops_cp.pop('type')
        if hptype in ('int', 'float'):
            hp_low, hp_high = hpprops_cp.pop('range')
            do_log = hpprops_cp.pop('do_log')
            tsnfrm_dict = hpprops_cp.pop('transform')
            if tsnfrm_dict is not None:
                fw_tns_str = tsnfrm_dict['forward']
                bw_tns_str = tsnfrm_dict['backward']
            else:
                fw_tns_str, bw_tns_str = 'x', 'y'
        elif hptype in ('cat'):
            hp_choices = hpprops_cp.pop('choices')
            assert 'range' not in hpprops_cp, 'cat variables -> no range'
            assert 'do_log' not in hpprops_cp, 'cat variables -> no do_log'
            assert 'transform' not in hpprops_cp, 'cat variables -> no transform'
        else:
            raise RuntimeError(f'hyper-param type {hptype} not implemented')

        assert len(hpprops_cp) == 0, f'hp props {list(hpprops_cp.keys())} not implemented.'

        if hptype in ('int', 'float'):
            fw_tns = lambda x: eval(fw_tns_str)
            bw_tns = lambda y: eval(bw_tns_str)

            assert np.isclose(hp_low, bw_tns(fw_tns(hp_low))), f'{hp_low} != f_inv(f({hp_low}))'
            assert np.isclose(hp_high, bw_tns(fw_tns(hp_high))), f'{hp_high} != f_inv(f({hp_high}))'

            def forward_fun(hpvals, fw_tns_str_=fw_tns_str,
                            do_log_=(self_logcasting and do_log)):
                fw_tns = lambda x: eval(fw_tns_str_)
                hp_tnsvals = fw_tns(hpvals)
                if do_log_:
                    hp_tnsvals = np.log(hp_tnsvals)
                return hp_tnsvals

            def backward_fun(hp_tnsvals, bw_tns_str_=bw_tns_str,
                             do_exp_=(self_logcasting and do_log),
                             do_round_=(hptype=='int')):
                bw_tns = lambda y: eval(bw_tns_str_)
                if do_exp_:
                    hp_vals = np.exp(hp_tnsvals)
                hp_vals = bw_tns(hp_vals)
                if do_exp_ and do_round_:
                    hp_vals = np.round(hp_vals).astype(np.int)
                return hp_vals

            eff_hptype = hptype
        elif (hptype in ('cat',)) and self_logcasting:
            int_fn = lambda v: int(np.round(v))
            #int_fn = lambda v: int(v)
            def forward_fun(hpvals, choices=hp_choices):
                if isinstance(hpvals, (list, tuple, np.ndarray, pd.Series)):
                    out = [choices.index(val) for val in hpvals]
                elif isinstance(hpvals, str):
                    out = choices.index(hpvals)
                else:
                    raise ValueError(f'I dont know what to do with {type(hpvals)}')
                return out

            def backward_fun(hp_tnsvals, choices=hp_choices):
                if isinstance(hp_tnsvals, (list, tuple, np.ndarray, pd.Series)):
                    hp_tnsvals = [x-1 if x==len(hp_choices) else x
                                  for x in hp_tnsvals]
                    assert all(0<=x for x in hp_tnsvals)
                    assert all(x<len(hp_choices) for x in hp_tnsvals)
                    out = [choices[int_fn(idx)] for idx in hp_tnsvals]
                elif isinstance(hp_tnsvals, (int, float)):
                    hp_tnsvals = hp_tnsvals - int(hp_tnsvals == len(hp_choices))
                    assert hp_tnsvals <= hp_tnsvals < len(hp_choices)
                    out = choices[int_fn(hp_tnsvals)]
                else:
                    raise ValueError(f'I dont know what to do with {type(hp_tnsvals)}')
                return out

            hp_low, hp_high = hp_choices[0], hp_choices[-1]
            do_log = False
            eff_hptype = 'int'
        elif (hptype in ('cat',)) and not(self_logcasting):
            forward_fun = lambda x: x
            backward_fun = lambda y: y

            hp_low, hp_high = hp_choices[0], hp_choices[-1]
            do_log = False
            eff_hptype = 'cat'
        else:
            raise RuntimeError(f'hyper-param type {hptype} not implemented')

        hpprops['forward_fun'] = forward_fun
        hpprops['backward_fun'] = backward_fun
        hpprops['range'] = [hp_low, hp_high]
        hpprops['eff_type'] = eff_hptype
        hpprops['do_log'] = do_log

    return hpo_paramprops

def smry2hpperfdf(df, method_xcols, run2perf):
    extracols = ['performance']
    if query_curriter_only:
        extracols = ['location'] + extracols

    hp_perf_list = []
    for xvals, x_df in df.groupby(method_xcols):
        if not ASSERT_UNQ_LOC:
            aaa = x_df['location'].tolist()[0]
            x_df = x_df[x_df['location'] == aaa]
        my_locs = x_df['location'].unique()
        assert len(my_locs) == 1, 'xcols is incomplete'
        (step_strat, step_picker), perf_col = run2perf
        if step_strat == 'max':
            perf_val = x_df.loc[[x_df[step_picker].idxmax()]][perf_col].values.item()
        elif step_strat == 'min':
            perf_val = x_df.loc[[x_df[step_picker].idxmin()]][perf_col].values.item()
        else:
            raise RuntimeError(f'{perf_cri} not implemented.')
        hp_perf_d = {col: val for col, val in zip(method_xcols, xvals)}
        hp_perf_d['performance'] = perf_val
        if query_curriter_only:
            hp_perf_d['location'] = my_locs[0]
        hp_perf_list.append(hp_perf_d)
    if len(hp_perf_list) > 0:
        hp_perfs_df = pd.DataFrame(hp_perf_list)
        hp_perfs_df = hp_perfs_df[method_xcols+extracols]
    else:
        hp_perfs_df = pd.DataFrame([], columns=method_xcols+extracols)
    return hp_perfs_df

def reduce_rngseeds(df, rng_seeds, xcols, reduction, do_assertions=False, drop_incomplete=False):
    nonseed_xcols = [x for x in xcols if x != 'rng_seed']
    outdf_list = []
    # To filter non-required rng_seeds in case you had 100 seed results
    df = df[df['rng_seed'].isin(rng_seeds)]
    for grp_vals, grp_df in df.groupby(nonseed_xcols):
        are_rngseeds_full = ((grp_df.shape[0] == len(rng_seeds)) and
                             (set(grp_df['rng_seed']) == set(rng_seeds)))
        assert not(do_assertions) or are_rngseeds_full, grp_df
        if drop_incomplete and not(are_rngseeds_full):
            continue
        perf_vals = grp_df['performance']
        grpdf_rngseeds = grp_df['rng_seed']

        if reduction is None:
            reduced_zip = list(zip(grpdf_rngseeds, perf_vals))
        elif reduction == 'mean':
            reduced_zip = [(f'mean({rng_seeds})', perf_vals.mean())]
        elif reduction == 'max':
            reduced_zip = [(f'max({rng_seeds})', perf_vals.max())]
        elif reduction == 'min':
            reduced_zip = [(f'min({rng_seeds})', perf_vals.min())]
        elif reduction == 'median':
            reduced_zip = [(f'median({rng_seeds})', perf_vals.median())]
        else:
            raise ValueError(f'reduction {reduction} not implemented')

        for seed_, perfval_ in reduced_zip:
            outdf_list.append([seed_] + list(grp_vals) + [perfval_])

    outdf = pd.DataFrame(outdf_list, columns=['rng_seed']+nonseed_xcols+['performance'])
    return outdf

def get_closerows(df, pnt_dict, hpo_paramprops, cmp_in_fwspace=False, rtol=1e-05, atol=1e-08):
    close_rowidxs = []
    for row_idx, row in df.iterrows():
        is_miss = False
        for hpname, hpprops in hpo_paramprops.items():
            hptype = hpprops['type']
            srcval, dstval = pnt_dict[hpname], row[hpname]
            if cmp_in_fwspace:
                forward_fun = hpprops['forward_fun']
                srcval, dstval = forward_fun(srcval), forward_fun(dstval)
            is_miss = is_miss or (hptype in ('int', 'float')) and not(np.isclose(srcval, dstval, rtol=rtol, atol=atol))
            is_miss = is_miss or (hptype in ('cat',) and not(srcval == dstval))
        if not is_miss:
            close_rowidxs.append(row_idx)
    return df.loc[close_rowidxs, :]

def dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    shared_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o : (d1[o], d2[o]) for o in shared_keys if d1[o] != d2[o]}
    same = set(o for o in shared_keys if d1[o] == d2[o])
    return added, removed, modified, same

def filter_initdf(ovat_perfsdf, method, method_xcols, hpo_paramprops, template_cfg):
    optimized_hps = list(hpo_paramprops.keys())
    fixed_hps = [hpname for hpname in method_xcols
                 if (hpname not in optimized_hps + ['rng_seed'])]

    # removing the init ovat elements that do not match our hpo
    keep_idxs = (ovat_perfsdf['method'] == method)
    for hpname in fixed_hps:
        keep_idxs = keep_idxs & (ovat_perfsdf[hpname] == template_cfg[hpname])

    for hpname, props in hpo_paramprops.items():
        hptype = props['type']
        if hptype in ('int', 'float'):
            pbnd_low, pbnd_high = props['range']
            keep_idxs = keep_idxs & (ovat_perfsdf[hpname] >= pbnd_low)
            keep_idxs = keep_idxs & (ovat_perfsdf[hpname] <= pbnd_high)
        elif hptype in ('cat'):
            hp_choices = props['choices']
            keep_idxs = keep_idxs & (ovat_perfsdf[hpname].isin(hp_choices))
        else:
            raise RuntimeError(f'hyper-param type {eff_hptype} not implemented')
    ovat_perfsdf = ovat_perfsdf[keep_idxs]
    return ovat_perfsdf

def update_expnames_json(expnames_json, hporounds_wildcard, hpo_exp, method):
    resdir_expname_method = []
    if exists(expnames_json):
        with open(expnames_json) as f:
            resdir_expname_method = json.load(f, object_pairs_hook=odict)

    my_tup = [hporounds_wildcard, hpo_exp, method]
    if my_tup not in resdir_expname_method:
        resdir_expname_method.append(my_tup)
        with open(expnames_json, 'w') as fp:
            fp.write('[\n')
            for i, lst_ in enumerate(resdir_expname_method):
                comma_str = '' if (i == len(resdir_expname_method) - 1) else ','
                fp.write(f'  {json.dumps(lst_)}{comma_str}\n')
            fp.write(']\n')

def load_smry(hpo_exp, method, method_xcols):
    if smry_fmt == 'csv':
        hpo_smry_path = f'{smry_tbls_dir}/{hpo_exp}/{method}.csv'
    elif smry_fmt == 'h5':
        hpo_smry_path = f'{smry_tbls_dir}/{hpo_exp}.h5'
    else:
        raise ValueError(f'summary format {smry_fmt} not implemented.')

    hpo_smry_df = None

    if exists(hpo_smry_path):
        if smry_fmt == 'csv':
            hpo_smry_df = pd.read_csv(hpo_smry_path)
        elif smry_fmt == 'h5':
            try:
                hpo_smry_df = pd.read_hdf(hpo_smry_path, key=method)
            except KeyError:
                pass
        else:
            raise ValueError(f'summary format {smry_fmt} not implemented.')
    return hpo_smry_df

def optuna_runner(hpo_superparams, ovat_perfsdf, hpo_perfsdf, hpo_paramprops):
    hpo_superparams_cp = dict(hpo_superparams)
    hpo_smapler_name = hpo_superparams_cp.pop('sampler')
    hpo_seed = hpo_superparams_cp.pop('seed')
    hpo_nbatch = hpo_superparams_cp.pop('batch_size')
    hpo_niters = hpo_superparams_cp.pop('num_iters')
    assert len(hpo_superparams_cp) == 0, \
           f'Unknwon hpo_superparams {list(hpo_superparams_cp.keys())}'

    np.random.seed(hpo_seed)
    random.seed(hpo_seed)

    import optuna
    from optuna.samplers import TPESampler
    from optuna.exceptions import ExperimentalWarning
    import warnings

    assert optuna.__version__ == '2.10.0'
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)

    if hpo_smapler_name == 'TPE':
        hpo_sampler = TPESampler(seed=hpo_seed)
    else:
        raise ValueError(f'Unknown sampler {hpo_smapler_name}')

    study = optuna.create_study(sampler=hpo_sampler, direction="maximize")
    # Adding the ovat results
    for _, (ovat_rowidx, ovat_row) in enumerate(ovat_perfsdf.iterrows()):
        trial_params = odict()
        trial_dists = odict()

        for hpname, hpprops in hpo_paramprops.items():
            eff_hptype = hpprops['eff_type']
            hpfwfun = hpprops['forward_fun']
            trial_params[hpname] = hpfwfun(ovat_row[hpname])

            if eff_hptype in ('int', 'float'):
                do_log = hpprops['do_log']
                hp_low, hp_high = hpprops['range']
                if self_logcasting:
                    # Since we are in charge of applying (tns, log, casting), we can use uniform for all hps.
                    optuna_dist = optuna.distributions.UniformDistribution
                    trial_dists[hpname] = optuna_dist(hpfwfun(hp_low), hpfwfun(hp_high))
                else:
                    # We're out sorucing the log-taking and casting to optuna
                    optuna_dists = {('float', True): optuna.distributions.LogUniformDistribution,
                                    ('float', False): optuna.distributions.UniformDistribution,
                                    ('int', True): optuna.distributions.IntLogUniformDistribution,
                                    ('int', False): optuna.distributions.IntUniformDistribution}
                    optuna_dist = optuna_dists[(eff_hptype, do_log)]
                    trial_dists[hpname] = optuna_dist(hp_low, hp_high)
            elif eff_hptype in ('cat',):
                hp_choices = hpprops['choices']
                optuna_dist = optuna.distributions.CategoricalDistribution
                trial_dists[hpname] = optuna_dist(choices=hp_choices)
            else:
                raise RuntimeError(f'hyper-param type {eff_hptype} not implemented')

        optuna_trial = optuna.trial.create_trial(params=trial_params,
                                                 distributions=trial_dists,
                                                 value=ovat_row['performance'])
        study.add_trial(optuna_trial)

    # The actual HPO loop
    for hpo_iter in range(hpo_niters):
        # create batch
        trial_ids = []
        trial_pnts = []
        for _ in range(hpo_nbatch):
            trial = study.ask()
            trial_ids.append(trial.number)
            suggpnt_dict = odict()
            for hpname, hpprops in hpo_paramprops.items():
                eff_hptype = hpprops['eff_type']
                if eff_hptype in ('int', 'float'):
                    hp_low, hp_high = hpprops['range']
                    if self_logcasting:
                        # The backward function will take care of any log/exp and casting to int/float
                        hpfwfun = hpprops['forward_fun']
                        hpbwfun = hpprops['backward_fun']
                        hp_tnslow, hp_tnshigh = hpfwfun(hp_low), hpfwfun(hp_high)
                        hp_tnssugg = trial.suggest_float(hpname, hp_tnslow, hp_tnshigh, log=False)
                        hp_sugg = hpbwfun(hp_tnssugg)
                    elif eff_hptype == 'int':
                        hp_sugg = trial.suggest_int(hpname, hp_low, hp_high, log=hpprops['do_log'])
                    elif eff_hptype == 'float':
                        hp_sugg = trial.suggest_float(hpname, hp_low, hp_high, log=hpprops['do_log'])
                    else:
                        raise RuntimeError(f'case not implemented')
                elif eff_hptype in ('cat',):
                    hp_choices = hpprops['choices']
                    hp_sugg = trial.suggest_categorical(hpname, hp_choices)
                else:
                    raise RuntimeError(f'hyper-param type {eff_hptype} not implemented')
                suggpnt_dict[hpname] = hp_sugg
            trial_pnts.append(suggpnt_dict)

        # Looking in the past runs for the value
        trial_vals = []
        if query_curriter_only:
            hpo_perfsdf_q = hpo_perfsdf[hpo_perfsdf['hpo_iter'] == hpo_iter]
        else:
            hpo_perfsdf_q = hpo_perfsdf
        for pnt in trial_pnts:
            pnt_df = get_closerows(hpo_perfsdf_q, pnt, hpo_paramprops,
                                   cmp_in_fwspace=False, rtol=1e-15, atol=1e-15)
            trial_vals.append(pnt_df['performance'].tolist())

        do_break = False
        for trial_id, pnt, val_list in zip(trial_ids, trial_pnts, trial_vals):
            if len(val_list)==0:
                msg_  = f'Stopping at HPO iteration {hpo_iter}.\n'
                msg_ += f'The following point (id={trial_id}) could not be found:\n'
                for k, v in pnt.items():
                    msg_ += f'  -> {k} = {v}\n'
                print(msg_)
                do_break = True
                break

        if do_break:
            break

        # finish all trials in the batch
        for trial_id, val_list in zip(trial_ids, trial_vals):
            msg_  = f'Found more than one value for a point. This is a problem for TPE.\n'
            msg_ += f'This could either be \n'
            msg_ += f'  (1) the result of improper rng_seed redduction, or \n'
            msg_ += f'  (2) some inaccucy when comparing and looking for the '
            msg_ += f'      point in the collected results.'
            assert not(hpo_smapler_name == 'TPE') or (len(val_list) <= 1), msg_
            for val in val_list:
                study.tell(trial_id, val)

    print(f'Best value so far: {study.best_value}')
    outdict = dict(next_pntsdict = trial_pnts,
                   next_iternum = hpo_iter,
                   is_hpo_done = not(do_break))

    return outdict

def bo_runner(hpo_superparams, ovat_perfsdf, hpo_perfsdf, hpo_paramprops):
    hpo_superparams_cp = dict(hpo_superparams)
    hpo_seed = hpo_superparams_cp.pop('seed')
    hpo_nbatch = hpo_superparams_cp.pop('batch_size')
    hpo_ninitpnts = hpo_superparams_cp.pop('init_points')
    hpo_niters = hpo_superparams_cp.pop('n_iter')
    hpo_acq = hpo_superparams_cp.pop('acq')
    hpo_kappa = hpo_superparams_cp.pop('kappa')
    hpo_kappadecay = hpo_superparams_cp.pop('kappa_decay')
    hpo_kappadecaydelay = hpo_superparams_cp.pop('kappa_decay_delay')
    hpo_xi = hpo_superparams_cp.pop('xi')
    assert len(hpo_superparams_cp) == 0, \
           f'Unknwon hpo_superparams {list(hpo_superparams_cp.keys())}'
    gp_kernel = 'matern'
    np.random.seed(hpo_seed)
    random.seed(hpo_seed)

    from bayes_opt import BayesianOptimization, UtilityFunction

    pbounds = odict()
    assert all(hpprops['eff_type'] in ('int', 'float') for
               hpprops in hpo_paramprops.values()), f'non-numer vars are not supported in BO!'
    assert self_logcasting, f'non-unif/log var priors are not implemented for BO!'
    pbounds = odict([(hpname, tuple(map(hpprops['forward_fun'], hpprops['range'])))
                     for hpname, hpprops in hpo_paramprops.items()])

    optimizer = BayesianOptimization(f=None, pbounds=pbounds, verbose=2, random_state=hpo_seed)
    utility = UtilityFunction(kind=hpo_acq, kappa=hpo_kappa, xi=hpo_xi,
                              kappa_decay=hpo_kappadecay,
                              kappa_decay_delay=hpo_kappadecaydelay)

    # Applying any OVAT initial results
    for _, (ovat_rowidx, ovat_row) in enumerate(ovat_perfsdf.iterrows()):
        pnt = odict()
        for hpname, hpprops in hpo_paramprops.items():
            hpfwfun = hpprops['forward_fun']
            pnt[hpname] = hpfwfun(ovat_row[hpname])
        optimizer.register(params=pnt, target=ovat_row['performance'])

    samplers = [lambda: optimizer.space.array_to_params(optimizer.space.random_sample())] * hpo_ninitpnts
    samplers = samplers + [lambda : optimizer.suggest(utility) for _ in range(hpo_nbatch)] * hpo_niters

    for hpo_iter, pnt_sampler in enumerate(samplers):
        print(f'HPO Iteration {hpo_iter}')
        trial_pnts_tns = [pnt_sampler() for _ in range(hpo_nbatch)]

        # Applying the backward function to go back to the original space
        trial_pnts = []
        for pnt_tns in trial_pnts_tns:
            pnt = odict()
            for hpname, hpprops in hpo_paramprops.items():
                hpbwfun = hpprops['backward_fun']
                hp_tnssugg = pnt_tns[hpname]
                pnt[hpname] = hpbwfun(hp_tnssugg)
            trial_pnts.append(pnt)

        # Looking in the past runs for the value
        trial_vals = []
        if query_curriter_only:
            hpo_perfsdf_q = hpo_perfsdf[hpo_perfsdf['hpo_iter'] == hpo_iter]
        else:
            hpo_perfsdf_q = hpo_perfsdf
        for pnt in trial_pnts:
            pnt_df = get_closerows(hpo_perfsdf_q, pnt, hpo_paramprops,
                                   cmp_in_fwspace=False, rtol=1e-15, atol=1e-15)
            trial_vals.append(pnt_df['performance'].tolist())

        do_break = False
        for pnt_idx, (pnt, val_list) in enumerate(zip(trial_pnts, trial_vals)):
            if len(val_list)==0:
                msg_  = f'Stopping at HPO iteration {hpo_iter}.\n'
                msg_ += f'The following point (round {hpo_iter} point {pnt_idx}) could not be found:\n'
                for k, v in pnt.items():
                    msg_ += f'  -> {k} = {v}\n'
                print(msg_)
                do_break = True
                break

        if do_break:
            break

        # finish all trials in the batch
        for pnt_idx, (pnt_tns, val_list) in enumerate(zip(trial_pnts_tns, trial_vals)):
            msg_  = f'Found more than one value for a point. This is a problem for Matern kernel.\n'
            msg_ += f'This could either be \n'
            msg_ += f'  (1) the result of improper rng_seed redduction, or \n'
            msg_ += f'  (2) some inaccucy when comparing and looking for the '
            msg_ += f'      point in the collected results.'
            #assert not(gp_kernel == 'matern') or (len(val_list) <= 1), msg_
            if (gp_kernel == 'matern') and (len(val_list) > 1):
                print(f'**********Warning***********\n{msg_}')
            optimizer.register(params=pnt_tns, target=val_list[0])
            #for val in val_list:
            #    optimizer.register(params=pnt_tns, target=val)

    print(f'Best target so far {optimizer.max["target"]}')
    outdict = dict(next_pntsdict = trial_pnts,
                   next_iternum = hpo_iter,
                   is_hpo_done = not(do_break))

    return outdict

def skopt_runner(hpo_superparams, ovat_perfsdf, hpo_perfsdf, hpo_paramprops):
    hpo_superparams_cp = dict(hpo_superparams)
    hpo_seed = hpo_superparams_cp.pop('seed')
    hpo_nbatch = hpo_superparams_cp.pop('batch_size')
    hpo_niters = hpo_superparams_cp.pop('n_iter')
    hpo_baseest = hpo_superparams_cp.pop('base_estimator')
    hpo_ninitpnts = hpo_superparams_cp.pop('n_initial_points')
    hpo_initpntsgen = hpo_superparams_cp.pop('initial_point_generator')
    hpo_njobs = hpo_superparams_cp.pop('n_jobs')
    hpo_acqfun = hpo_superparams_cp.pop('acq_func')
    hpo_acqopt = hpo_superparams_cp.pop('acq_optimizer')
    hpo_acqfunargs = hpo_superparams_cp.pop('acq_func_kwargs')
    hpo_acqoptargs = hpo_superparams_cp.pop('acq_optimizer_kwargs')
    cache_dir = hpo_superparams_cp.pop('cache_dir', None)
    do_cache_optimizer = hpo_superparams_cp.pop('do_cache_optimizer', False)
    assert len(hpo_superparams_cp) == 0, \
           f'Unknwon hpo_superparams {list(hpo_superparams_cp.keys())}'
    gp_kernel = 'white'
    np.random.seed(hpo_seed)
    random.seed(hpo_seed)

    assert not(do_cache_optimizer) or isinstance(cache_dir, str)
    assert not(do_cache_optimizer) or exists(cache_dir)
    assert not(do_cache_optimizer) or os.path.isdir(cache_dir)

    import skopt
    from skopt.space.space import Real, Integer, Categorical
    if cache_dir is not None:
        import dill as pickle

    assert all(hpprops['eff_type'] in ('int', 'float') for
               hpprops in hpo_paramprops.values()), f'non-numer vars are not supported in skopt!'
    assert self_logcasting, f'non-unif/log var priors are not implemented for skopt!'
    # Note: I think it is safe to remove the previous two assertions, but I never
    #       verified the functionality when these assertions are not imposed.

    dimensions = []
    for hpname, hpprops in hpo_paramprops.items():
        eff_hptype = hpprops['eff_type']
        hpfwfun = hpprops['forward_fun']
        if eff_hptype in ('int', 'float'):
            hp_low, hp_high = hpprops['range']
            do_log = hpprops['do_log']
            if self_logcasting:
                # Since we are in charge of applying (tns, log, casting), we can use uniform for all hps.
                skdim = Real(hpfwfun(hp_low), hpfwfun(hp_high), prior="uniform", base=10,
                             transform=None, name=hpname, dtype=np.float)
            else:
                # We're out sorucing the log-taking and casting to optuna
                dimmaker, dimdtype = {'float': (Real, np.float), 'int': (Integer, np.int64)}[eff_hptype]
                dimprior = "log-uniform" if do_log else "uniform"
                skdim = dimmaker(hp_low, hp_high, prior=dimprior, base=10,
                                 transform=None, name=hpname, dtype=dimdtype)
        elif eff_hptype in ('cat',):
            hp_choices = hpprops['choices']
            skdim = Categorical(hp_choices, prior=None, transform=None, name=hpname)
        else:
            raise RuntimeError(f'hyper-param type {eff_hptype} not implemented')
        dimensions.append(skdim)

    optimizer = skopt.Optimizer(dimensions, base_estimator=hpo_baseest,
                                n_initial_points=hpo_ninitpnts,
                                initial_point_generator=hpo_initpntsgen,
                                n_jobs=hpo_njobs, acq_func=hpo_acqfun,
                                acq_optimizer=hpo_acqopt, random_state=hpo_seed,
                                model_queue_size=None, acq_func_kwargs=hpo_acqfunargs,
                                acq_optimizer_kwargs=hpo_acqoptargs)

    if cache_dir is None:
        # No caching whatsoever!
        do_apply_ovat = True
        hpo_iterlist = list(range(hpo_niters))
        pklloadpath = None
    else:
        all_pkls = [f'{cache_dir}/init.pkl']
        all_pkls = all_pkls + [f'{cache_dir}/iter_{i}.pkl' for i in range(hpo_niters)]
        all_pkls_exist = [exists(pklpath) for pklpath in all_pkls]

        next_incomplete_idx = -1
        if any(all_pkls_exist):
            next_incomplete_idx = max(i for i, v in enumerate(all_pkls_exist) if (v == True))
        next_incomplete_idx += 1
        assert next_incomplete_idx >= 0

        do_apply_ovat = (next_incomplete_idx == 0)
        hpo_iterlist = list(range(max(0, next_incomplete_idx-1), hpo_niters))

        pklloadpath = None
        if (len(hpo_iterlist) > 0) and not(do_apply_ovat):
            pklloadpath = all_pkls[hpo_iterlist[0]]

    if do_apply_ovat:
        # Applying any OVAT initial results
        ovat_pnts, ovat_trgs = [], []
        for _, (ovat_rowidx, ovat_row) in enumerate(ovat_perfsdf.iterrows()):
            pnt = [hpprops['forward_fun'](ovat_row[hpname])
                   for _, (hpname, hpprops) in enumerate(hpo_paramprops.items())]
            ovat_pnts.append(pnt)
            ovat_trgs.append(-ovat_row['performance'])
            # The negative is since that skopt minimizes

        if len(ovat_pnts):
            optimizer.tell(x=ovat_pnts, y=ovat_trgs)

        if cache_dir is not None:
            print(f'SKOPT CACHING: Writing to pickle file: {cache_dir}/init.pkl')
            with open(f'{cache_dir}/init.pkl', 'wb') as handle:
                 pickle.dump(optimizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if pklloadpath is not None:
        print(f'SKOPT CACHING: Loading from pickle file: {pklloadpath}')
        with open(pklloadpath, 'rb') as handle:
            optimizer = pickle.load(handle)

    # The actual HPO loop
    for hpo_iter in hpo_iterlist:
        print(f'HPO Iteration {hpo_iter}')
        trial_pnts_tns = optimizer.ask(n_points=hpo_nbatch, strategy='cl_min')

        # Applying the backward function to go back to the original space
        trial_pnts = [odict([(hpname, hpprops['backward_fun'](pnt_tns[hpidx]))
                             for hpidx, (hpname, hpprops)
                             in enumerate(hpo_paramprops.items())])
                      for pnt_tns in trial_pnts_tns]

        # Looking in the past runs for the value
        trial_vals = []
        if query_curriter_only:
            hpo_perfsdf_q = hpo_perfsdf[hpo_perfsdf['hpo_iter'] == hpo_iter]
        else:
            hpo_perfsdf_q = hpo_perfsdf
        for pnt in trial_pnts:
            pnt_df = get_closerows(hpo_perfsdf_q, pnt, hpo_paramprops,
                                   cmp_in_fwspace=False, rtol=1e-12, atol=1e-12)
            trial_vals.append((-pnt_df['performance']).tolist()) # The negative is since that skopt minimizes

        do_break = False
        for pnt_idx, (pnt, val_list) in enumerate(zip(trial_pnts, trial_vals)):
            if len(val_list)==0:
                msg_  = f'Stopping at HPO iteration {hpo_iter}.\n'
                msg_ += f'The following point (round {hpo_iter} point {pnt_idx}) could not be found:\n'
                for k, v in pnt.items():
                    msg_ += f'  -> {k} = {v}\n'
                print(msg_)
                do_break = True
                break

        if do_break:
            break

        # finish all trials in the batch
        for pnt_idx, (pnt_tns, val_list) in enumerate(zip(trial_pnts_tns, trial_vals)):
            msg_  = f'Found more than one value for a point. This is a problem for Matern kernel.\n'
            msg_ += f'This could either be \n'
            msg_ += f'  (1) the result of improper rng_seed redduction, or \n'
            msg_ += f'  (2) some inaccucy when comparing and looking for the '
            msg_ += f'      point in the collected results.'
            assert not(gp_kernel == 'white') or (len(val_list) <= 1), msg_
            for val in val_list:
                optimizer.tell(x=pnt_tns, y=val)

        if cache_dir is not None:
            print(f'SKOPT CACHING: Writing to pickle file: {cache_dir}/iter_{hpo_iter}.pkl')
            with open(f'{cache_dir}/iter_{hpo_iter}.pkl', 'wb') as handle:
                 pickle.dump(optimizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if len(optimizer.yi) > 0:
        print(f'Best target so far {-min(optimizer.yi)}')
    outdict = dict(next_pntsdict = trial_pnts,
                   next_iternum = hpo_iter,
                   is_hpo_done = not(do_break))

    return outdict

def gpoyopt_runner(hpo_superparams, ovat_perfsdf, hpo_perfsdf, hpo_paramprops):
    hpo_superparams_cp = dict(hpo_superparams)
    hpo_seed = hpo_superparams_cp.pop('seed')
    hpo_nbatch = hpo_superparams_cp.pop('batch_size')
    hpo_niters = hpo_superparams_cp.pop('n_iter')

    hpo_ninitpnts = hpo_superparams_cp.pop('initial_design_numdata')
    hpo_evaltype = hpo_superparams_cp.pop('evaluator_type')
    hpo_acqfun = hpo_superparams_cp.pop('acquisition_type')
    hpo_normtarg = hpo_superparams_cp.pop('normalize_Y')
    hpo_isfexact = hpo_superparams_cp.pop('exact_feval')
    hpo_mdltype = hpo_superparams_cp.pop('model_type')
    hpo_initpntsgen = hpo_superparams_cp.pop('initial_design_type')
    hpo_acqopt = hpo_superparams_cp.pop('acquisition_optimizer_type')
    hpo_updtintvl = hpo_superparams_cp.pop('model_update_interval')
    hpo_ncores = hpo_superparams_cp.pop('num_cores')
    hpo_dedup = hpo_superparams_cp.pop('de_duplication')

    assert len(hpo_superparams_cp) == 0, \
           f'Unknwon hpo_superparams {list(hpo_superparams_cp.keys())}'
    gp_kernel = 'white'
    np.random.seed(hpo_seed)
    random.seed(hpo_seed)

    import GPyOpt
    from GPyOpt.methods import BayesianOptimization

    assert all(hpprops['eff_type'] in ('int', 'float') for
               hpprops in hpo_paramprops.values()), f'non-numer vars are not supported in GPyOpt!'
    assert self_logcasting, f'non-unif/log var priors are not implemented for GPyOpt!'

    assert not(hpo_nbatch > 1) or (hpo_evaltype != 'sequential')

    domain = [{'name': hpname, 'type': 'continuous',
               'domain': tuple(map(hpprops['forward_fun'], hpprops['range']))}
              for hpname, hpprops in hpo_paramprops.items()]

    # Applying any OVAT initial results
    ovat_pnts, ovat_trgs = [], []
    for _, (ovat_rowidx, ovat_row) in enumerate(ovat_perfsdf.iterrows()):
        pnt = [hpprops['forward_fun'](ovat_row[hpname])
               for _, (hpname, hpprops) in enumerate(hpo_paramprops.items())]
        ovat_pnts.append(pnt)
        ovat_trgs.append(ovat_row['performance'])
    if len(ovat_pnts):
        X_sofar = np.array(ovat_pnts).reshape(-1, len(hpo_paramprops))
        Y_sofar = np.array(ovat_trgs).reshape(-1, 1)
    else:
        X_sofar, Y_sofar = None, None

    for hpo_iter in range(hpo_niters):
        print(f'HPO Iteration {hpo_iter}')
        if (X_sofar is None) or (Y_sofar is None):
            from GPyOpt.core.task.space import Design_space
            from GPyOpt.experiment_design import initial_design
            self_space_ = Design_space(domain, None)
            trial_pnts_tns = initial_design(hpo_initpntsgen, self_space_,
                                            hpo_ninitpnts)
        else:
            bo_step = BayesianOptimization(f=None, domain=domain,
                                           X=X_sofar, Y=Y_sofar,
                                           batch_size=hpo_nbatch,
                                           initial_design_numdata=hpo_ninitpnts,
                                           evaluator_type=hpo_evaltype,
                                           acquisition_type=hpo_acqfun,
                                           normalize_Y=hpo_normtarg,
                                           exact_feval=hpo_isfexact,
                                           model_type=hpo_mdltype,
                                           initial_design_type=hpo_initpntsgen,
                                           acquisition_optimizer_type=hpo_acqopt,
                                           model_update_interval=hpo_updtintvl,
                                           num_cores=hpo_ncores,
                                           maximize=True,
                                           de_duplication=hpo_dedup)

            trial_pnts_tns = bo_step.suggest_next_locations()

        # Applying the backward function to go back to the original space
        trial_pnts = [odict([(hpname, hpprops['backward_fun'](pnt_tns[hpidx]))
                             for hpidx, (hpname, hpprops)
                             in enumerate(hpo_paramprops.items())])
                      for pnt_tns in trial_pnts_tns]

        # Looking in the past runs for the value
        trial_vals = []
        if query_curriter_only:
            hpo_perfsdf_q = hpo_perfsdf[hpo_perfsdf['hpo_iter'] == hpo_iter]
        else:
            hpo_perfsdf_q = hpo_perfsdf
        for pnt in trial_pnts:
            pnt_df = get_closerows(hpo_perfsdf_q, pnt, hpo_paramprops,
                                   cmp_in_fwspace=False, rtol=1e-15, atol=1e-15)
            trial_vals.append((pnt_df['performance']).tolist())

        do_break = False
        for pnt_idx, (pnt, val_list) in enumerate(zip(trial_pnts, trial_vals)):
            if len(val_list)==0:
                msg_  = f'Stopping at HPO iteration {hpo_iter}.\n'
                msg_ += f'The following point (round {hpo_iter} point {pnt_idx}) could not be found:\n'
                for k, v in pnt.items():
                    msg_ += f'  -> {k} = {v}\n'
                print(msg_)
                do_break = True
                break

        if do_break:
            break

        # finish all trials in the batch
        x_sofar_list = [X_sofar] if (X_sofar is not None) else []
        y_sofar_list = [Y_sofar] if (Y_sofar is not None) else []
        assert len(x_sofar_list) == len(y_sofar_list)
        for pnt_idx, (pnt_tns, val_list) in enumerate(zip(trial_pnts_tns, trial_vals)):
            msg_  = f'Found more than one value for a point. This is a problem for Matern kernel.\n'
            msg_ += f'This could either be \n'
            msg_ += f'  (1) the result of improper rng_seed redduction, or \n'
            msg_ += f'  (2) some inaccucy when comparing and looking for the '
            msg_ += f'      point in the collected results.'
            assert not(hpo_isfexact) or (len(val_list) <= 1), msg_
            for val in val_list:
                x_sofar_list.append(pnt_tns.reshape(1, len(hpo_paramprops)))
                y_sofar_list.append(np.array([[val]]))
        X_sofar = np.concatenate(x_sofar_list, axis=0)
        Y_sofar = np.concatenate(y_sofar_list, axis=0)

    if (Y_sofar is not None) and (len(Y_sofar) > 0):
        print(f'Best target so far {Y_sofar.max()}')
    outdict = dict(next_pntsdict = trial_pnts,
                   next_iternum = hpo_iter,
                   is_hpo_done = not(do_break))

    return outdict

def prosrs_runner(hpo_superparams, ovat_perfsdf, hpo_perfsdf, hpo_paramprops):
    hpo_superparams_cp = dict(hpo_superparams)
    hpo_seed = hpo_superparams_cp.pop('seed')
    hpo_nbatch = hpo_superparams_cp.pop('batch_size')
    hpo_niters = hpo_superparams_cp.pop('n_iter')

    hpo_nitersprosrs = hpo_superparams_cp.pop('n_iter_prosrs')
    hpo_ncycle = hpo_superparams_cp.pop('n_cycle')
    hpo_niterdoe = hpo_superparams_cp.pop('n_iter_doe')
    hpo_partrain = hpo_superparams_cp.pop('parallel_training')

    assert len(hpo_superparams_cp) == 0, \
           f'Unknwon hpo_superparams {list(hpo_superparams_cp.keys())}'
    gp_kernel = 'white'
    np.random.seed(hpo_seed)
    random.seed(hpo_seed)

    import prosrs
    from prosrs import Optimizer
    import tempfile
    import shutil

    assert all(hpprops['eff_type'] in ('int', 'float') for
               hpprops in hpo_paramprops.values()), f'non-numer vars are not supported in GPyOpt!'
    assert self_logcasting, f'non-unif/log var priors are not implemented for GPyOpt!'

    domain = [tuple(map(hpprops['forward_fun'], hpprops['range']))
              for hpname, hpprops in hpo_paramprops.items()]

    out_dir_prosrs = tempfile.mkdtemp()
    prob = prosrs.Problem(domain, name='prosrs_hpo',
                          x_var=list(hpo_paramprops.keys()),
                          y_var='performance')

    optimizer = Optimizer(prob, hpo_nbatch, n_iter=hpo_nitersprosrs,
                          n_iter_doe=hpo_niterdoe,
                          n_cycle=hpo_ncycle, resume=False,
                          seed=hpo_seed, seed_func=None,
                          parallel_training=hpo_partrain,
                          out_dir=out_dir_prosrs)

    # Applying any OVAT initial results
    ovat_pnts, ovat_trgs = [], []
    for _, (ovat_rowidx, ovat_row) in enumerate(ovat_perfsdf.iterrows()):
        pnt = [hpprops['forward_fun'](ovat_row[hpname])
               for _, (hpname, hpprops) in enumerate(hpo_paramprops.items())]
        ovat_pnts.append(pnt)
        ovat_trgs.append(-ovat_row['performance']) # The negative is since that ProSRS minimizes

    if len(ovat_pnts) > 0:
        X_ovat = np.array(ovat_pnts).reshape(-1, len(hpo_paramprops))
        Y_ovat = np.array(ovat_trgs).reshape(-1)
        optimizer.gSRS_pct = np.nan
        optimizer.t_build = np.nan
        optimizer.t_srs = np.nan
        optimizer.t_prop = 0.0
        optimizer.update(X_ovat, Y_ovat, verbose=True)


    for hpo_iter in range(hpo_niters):
        print(f'HPO Iteration {hpo_iter}')
        trial_pnts_tns = optimizer.propose(verbose=True)

        # Applying the backward function to go back to the original space
        trial_pnts = [odict([(hpname, hpprops['backward_fun'](pnt_tns[hpidx]))
                             for hpidx, (hpname, hpprops)
                             in enumerate(hpo_paramprops.items())])
                      for pnt_tns in trial_pnts_tns]

        # Looking in the past runs for the value
        trial_vals = []
        if query_curriter_only:
            hpo_perfsdf_q = hpo_perfsdf[hpo_perfsdf['hpo_iter'] == hpo_iter]
        else:
            hpo_perfsdf_q = hpo_perfsdf
        for pnt in trial_pnts:
            pnt_df = get_closerows(hpo_perfsdf_q, pnt, hpo_paramprops,
                                   cmp_in_fwspace=False, rtol=1e-15, atol=1e-15)
            trial_vals.append((-pnt_df['performance']).tolist()) # The negative is since that ProSRS minimizes

        do_break = False
        for pnt_idx, (pnt, val_list) in enumerate(zip(trial_pnts, trial_vals)):
            if len(val_list)==0:
                msg_  = f'Stopping at HPO iteration {hpo_iter}.\n'
                msg_ += f'The following point (round {hpo_iter} point {pnt_idx}) could not be found:\n'
                for k, v in pnt.items():
                    msg_ += f'  -> {k} = {v}\n'
                print(msg_)
                do_break = True
                break

        if do_break:
            break

        # finish all trials in the batch
        x_new_list, y_new_list = [], []
        for pnt_idx, (pnt_tns, val_list) in enumerate(zip(trial_pnts_tns, trial_vals)):
            msg_  = f'Found more than one value for a point. This is a problem for Matern kernel.\n'
            msg_ += f'This could either be \n'
            msg_ += f'  (1) the result of improper rng_seed redduction, or \n'
            msg_ += f'  (2) some inaccucy when comparing and looking for the '
            msg_ += f'      point in the collected results.'
            if (len(val_list) > 1):
                print(f'*************{Warning}**********:\n {msg_}')
            for val in val_list:
                x_new_list.append(pnt_tns.reshape(1, len(hpo_paramprops)))
                y_new_list.append(np.array([val]))
        X_new = np.concatenate(x_new_list, axis=0)
        Y_new = np.concatenate(y_new_list, axis=0)
        optimizer.update(X_new, Y_new, verbose=False)

    print(f'Best target so far {-optimizer.best_y}')
    #optimizer.show(select=['result'], n_top=1)

    outdict = dict(next_pntsdict = trial_pnts,
                   next_iternum = hpo_iter,
                   is_hpo_done = not(do_break))

    if exists(out_dir_prosrs):
        shutil.rmtree(out_dir_prosrs)

    return outdict

def itemer(v):
    if isinstance(v, (str, int, float)):
        o = v
    elif isinstance(v, np.integer):
        o = int(v)
    elif isinstance(v, np.floating):
        o = float(v)
    else:
        o = v.item()
    return o

def write_nextiter_config(next_pntsdict, template_cfg, outcfg_path, zip_paramnames):
    # Writing the next rounds config to the disk
    nextrnd_cfgdict = copy.deepcopy(template_cfg)

    for hpname in zip_paramnames:
        nextrnd_cfgdict[hpname] = [itemer(pnt[hpname]) for pnt in next_pntsdict]
        nextrnd_cfgdict['looping_tree'][hpname] = 'zip'

    # import os
    # if exists(outcfg_path):
    #     os.remove(f'{outcfg_path}')

    if exists(outcfg_path):
        with open(outcfg_path) as f:
            nextrnd_cfgdict_disk = json.load(f, object_pairs_hook=odict)

        if nextrnd_cfgdict != nextrnd_cfgdict_disk:
            ad, rm, md, same = dict_compare(nextrnd_cfgdict, nextrnd_cfgdict_disk)
            msg_  = f'There is an existing config at {outcfg_path} '
            msg_ += f'that does not match my expectation:\n'
            msg_ += f'  -> {list(ad)} are keys that do not exist on '
            msg_ += f'the disk, but does exist in my config.\n'
            msg_ += f'  -> {list(rm)} are keys that do exist on the disk, '
            msg_ += f'but do not exist in my config.\n'
            msg_ += f'  -> {list(md.keys())} are the non-equal keys.\n'
            msg_ += '-'*40 + '\n'
            msg_ += f'Please resolve the issue, and remove any '
            msg_ += f'non-matching residual runs/configs carefully.'
            raise RuntimeError(msg_)
    else:
        if not(only_dryrun):
            with open(outcfg_path, 'w') as fp:
                json.dump(nextrnd_cfgdict, fp, indent=2)

    print(f'Please run "{outcfg_path}", and come back for the next iteration of HPO!')

def main(hpocfg_treeid):
    # Unfortunately, the hash seed is necessary to
    # fix for all libraries reproducibility.
    ensure_pythonhashseed(seed=0, failure_action='raise_error')
    np.random.seed(0)
    random.seed(0)

    hpocfg_treeid_split = hpocfg_treeid.split('/')
    hpocfg_tree = '/'.join(hpocfg_treeid_split[:-1])
    hpocfg_id = hpocfg_treeid_split[-1]
    hpo_cfgpath = f'{PROJPATH}/configs/{hpocfg_tree}/{hpocfg_id}.json'
    expnames_json = f'{PROJPATH}/configs/expnames.json'

    # Reading the HPO configuration json file
    with open(hpo_cfgpath) as f:
        hpo_cfgdict = json.load(f, object_pairs_hook=odict)
    hpo_lib = hpo_cfgdict.pop('library')
    hpo_paramprops = hpo_cfgdict.pop('param_props')
    hpo_iternames_wc = hpo_cfgdict.pop('hpo_iternames')
    hpo_initexp = hpo_cfgdict.pop('init_experiment')
    hpo_exp = hpo_cfgdict.pop('hpo_experiment')
    hpo_superparams = hpo_cfgdict.pop('hpo_superparams')
    template_cfg = hpo_cfgdict.pop('template_cfg')
    run2perf = hpo_cfgdict.pop('run2perf')
    rngseed_reduction = hpo_cfgdict.pop('rngseed_reduction')
    pick_random_rngseeds = hpo_cfgdict.pop('pick_random_rngseeds', False)
    assert len(hpo_cfgdict) == 0, f'Unknown keys {hpo_cfgdict.keys()}'
    method = template_cfg['method']
    envname = template_cfg['environment']
    method_xcols = xcols[method]

    # Making sure template_cfg is fine
    for xcol in method_xcols:
        if xcol not in ('num_envs', 'rng_seed'):
            assert template_cfg['looping_tree'][xcol] == 'fixed', xcol
    assert template_cfg['looping_tree']['method'] == 'fixed'
    assert template_cfg['looping_tree']['environment'] == 'fixed'
    assert template_cfg['looping_tree']['rng_seed'] == 'cartesian'
    assert not(pick_random_rngseeds) or (hpo_lib in ('skopt', 'GPyOpt', 'prosrs'))
    if (hpo_lib=='skopt') and (mpi_rank > 1):
        print('Warning: Using MPI with skopt slows it down heavily for some odd reason.')

    # Amending paramprops
    hpo_paramprops = amend_paramprops(hpo_paramprops)

    # Making sure that the proper experiment names are declared in the global expnames.json
    hporounds_wildcard = f'{hpocfg_tree}/{hpo_iternames_wc}'
    if mpi_rank == 0:
        update_expnames_json(expnames_json, hporounds_wildcard, hpo_exp, method)
    # Making sure that the summary files are uptodate with the most recent runs
    if not(only_dryrun) and not(hpo_lib=='skopt'):
        csv2summ_main(use_mpi=True, be_lazy=True)
    if mpi_rank > 0:
        return None

    # Reading the initial ovat df
    initsmrydf = load_smry(hpo_initexp, method, method_xcols)
    if initsmrydf is None:
        initsmrydf = pd.DataFrame([], columns=method_xcols+['performance'])
    initsmrydf = filter_initdf(initsmrydf, method, method_xcols, hpo_paramprops, template_cfg)

    # Reading the hpo iteration dfs
    hpo_smry_df = load_smry(hpo_exp, method, method_xcols)
    if hpo_smry_df is None:
        hpo_smry_df = pd.DataFrame([], columns=method_xcols+['performance'])

    # Converting summaries to performance data frames
    ovat_perfsdf_seeded = smry2hpperfdf(initsmrydf, method_xcols, run2perf)
    hpo_perfsdf_seeded = smry2hpperfdf(hpo_smry_df, method_xcols, run2perf)

    # TODO: Improve the iteration filtering!
    if query_curriter_only:
        a = {hporounds_wildcard.replace('*', str(i)): i for i in range(1000)}
        hpo_perfsdf_seeded['hpo_iter'] = [a.get('/'.join(x.split('/')[:-1]), None)
                                          for x in hpo_perfsdf_seeded['location']]
        isnl = hpo_perfsdf_seeded['hpo_iter'].isnull()
        hpo_perfsdf_seeded = hpo_perfsdf_seeded[~isnl].copy(deep=True)
        hpo_perfsdf_seeded['hpo_iter'] = hpo_perfsdf_seeded['hpo_iter'].astype(int)

    # Applying all the pre-transformations to the ovat data
    ovat_perfsdf = reduce_rngseeds(ovat_perfsdf_seeded, template_cfg['rng_seed'],
                                   method_xcols, rngseed_reduction,
                                   drop_incomplete=not(pick_random_rngseeds))

    # Applying all the pre-transformations to the hpo data
    alwed_cols = method_xcols+['hpo_iter'] if query_curriter_only else method_xcols
    hpo_perfsdf = reduce_rngseeds(hpo_perfsdf_seeded, template_cfg['rng_seed'],
                                  alwed_cols, rngseed_reduction,
                                  drop_incomplete=not(pick_random_rngseeds))

    if hpo_lib == 'optuna':
        hpo_outdict = optuna_runner(hpo_superparams, ovat_perfsdf, hpo_perfsdf, hpo_paramprops)
    elif hpo_lib == 'BayesianOptimization':
        hpo_outdict = bo_runner(hpo_superparams, ovat_perfsdf, hpo_perfsdf, hpo_paramprops)
    elif hpo_lib == 'skopt':
        do_cache_optimizer = hpo_superparams.get('do_cache_optimizer', False)
        if do_cache_optimizer:
            dflt_cache_dir = f'{PROJPATH}/trash/skopt_cache/{hpocfg_tree}/{hpocfg_id}'
            if hpo_superparams.get('cache_dir', None) is None:
                cache_dir = dflt_cache_dir
                os.makedirs(cache_dir, exist_ok=True)
        else:
            assert hpo_superparams.get('cache_dir', None) is None
            cache_dir = None
        hpo_superparams['cache_dir'] = cache_dir
        hpo_outdict = skopt_runner(hpo_superparams, ovat_perfsdf, hpo_perfsdf, hpo_paramprops)
    elif hpo_lib == 'GPyOpt':
        hpo_outdict = gpoyopt_runner(hpo_superparams, ovat_perfsdf, hpo_perfsdf, hpo_paramprops)
    elif hpo_lib == 'prosrs':
        hpo_outdict = prosrs_runner(hpo_superparams, ovat_perfsdf, hpo_perfsdf, hpo_paramprops)
    else:
        raise RuntimeError(f'library {hpo_libname} not implemented')

    # Retrieving the next next set of runs!
    next_pntsdict = hpo_outdict['next_pntsdict']
    next_iternum = hpo_outdict['next_iternum']
    is_hpo_done = hpo_outdict['is_hpo_done']

    # The next iteration json config file path
    nextrnd_cfgtreeid = hporounds_wildcard.replace("*", str(next_iternum))
    nextrnd_cfgpath = f'{PROJPATH}/configs/{nextrnd_cfgtreeid}.json'

    # Determining whether we need to randomize rng_seeds or do a cartesian product
    zip_paramnames = list(hpo_paramprops.keys())
    if pick_random_rngseeds:
        assert template_cfg['looping_tree']['rng_seed'] == 'cartesian'
        rngseed_choices = np.array(template_cfg['rng_seed'])
        n_rngseeds = len(rngseed_choices)
        np_random = np.random.RandomState(seed=next_iternum*100)
        for pnt_idx, pnt in enumerate(next_pntsdict):
            ii = int(np_random.randint(n_rngseeds))
            pnt['rng_seed'] = rngseed_choices[ii]
        zip_paramnames = ['rng_seed'] + zip_paramnames

    if is_hpo_done:
        print('HPO is successfully done!')
    else:
        # carefully writing the next config file
        write_nextiter_config(next_pntsdict, template_cfg, nextrnd_cfgpath, zip_paramnames)

if __name__ == '__main__':
    use_argparse = True
    if use_argparse:
        import argparse
        my_parser = argparse.ArgumentParser()
        my_parser.add_argument('-c', '--cfgtreeid', action='store', type=str, required=True)
        args = my_parser.parse_args()
        args_hpocfgtreeid = args.cfgtreeid
    else:
        args_hpocfgtreeid = '0_scratch/hpo_quicktest/trpo_hpo'

    main(args_hpocfgtreeid)
