import os
import numpy as np
import json
import pandas as pd
import time
from os.path import isdir, exists
from cfg import results_dir as proj_resdir
from cfg import resdir_expname_method, frozen_exps
from cfg import eval_xcols, eval_idcols, eval_ycols
from cfg import xcols, ycols, tcols, ignore_cols
from cfg import col_renames, smry_tbls_dir, smry_fmt
from collections import OrderedDict, defaultdict
from pathlib import Path
import shutil

def is_dir_all_older_than_file(dir_, file_):
    file_time_ = os.path.getmtime(file_)
    a = all(os.path.getmtime(os.path.join(root, subdirectory)) < file_time_
            for root, subdirectories, files in os.walk(dir_)
            for subdirectory in subdirectories)

    b = all(os.path.getmtime(os.path.join(root, file)) < file_time_
            for root, subdirectories, files in os.walk(dir_)
            for file in files)
    return (a and b)

def main(use_mpi=True, be_lazy=True):
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        mpi_rank = comm.Get_rank()
        mpi_size = comm.Get_size()
    else:
        mpi_rank, mpi_size = 0, 1

    assert smry_fmt in ('h5', 'csv')

    output_dir = smry_tbls_dir
    # Recreating the output directory in case it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    import random, string
    random_state_ = random.getstate()
    random.seed(None)
    rndstr=''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    tmp_dir = f'{output_dir}/temp_{rndstr}'
    random.setstate(random_state_)
    if use_mpi:
        tmp_dir = comm.bcast(tmp_dir)

    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    tmp_postfix_maker = lambda rank: f'r{rank}'

    # Collecting all directories to walk over
    method_cfgdirs = defaultdict(list)
    has_exp_changed = defaultdict(lambda: False)
    for resdir, expname, method in resdir_expname_method:
        cfgdirslist_ = os.listdir(f'{proj_resdir}/{resdir}')
        try:
            sort_fn = lambda x: int(x.split('_')[-1])
            cfgdirslist = sorted(cfgdirslist_, key=sort_fn)
        except Exception:
            cfgdirslist = sorted(cfgdirslist_)

        cfgresdir_expname = []
        for cfgdir in cfgdirslist:
            cfgresdir = f'{proj_resdir}/{resdir}/{cfgdir}'
            if isdir(cfgresdir):
                cfgresdir_expname.append( ((proj_resdir, resdir, cfgdir), expname) )
            if be_lazy:
                if smry_fmt == 'h5':
                    finalagg_out = f'{output_dir}/{expname}.h5'
                elif smry_fmt == 'csv':
                    finalagg_out = f'{output_dir}/{expname}/{method}.csv'
                else:
                    raise ValueError(f'{smry_fmt} not implemented')
                if exists(finalagg_out):
                    isolder = is_dir_all_older_than_file(cfgresdir, finalagg_out)
                else:
                    isolder = False
                has_exp_changed[expname] = has_exp_changed[expname] or not(isolder)
        method_cfgdirs[method] += cfgresdir_expname

    if be_lazy:
        if mpi_rank == 0:
            print(f'Laziness report:')
            for expname, has_changed in has_exp_changed.items():
                notstr = "" if has_changed else "not "
                print(f' {expname} has {notstr}changed since last summary.', flush=True)
        method_cfgdirs = {method: [(cfgresdir, expname)
                                   for (cfgresdir, expname) in cfgresdir_expnames
                                   if has_exp_changed[expname]]
                          for method, cfgresdir_expnames in method_cfgdirs.items()}

    # Making sure we're excluding the frozen experiments
    method_cfgdirs = {method: [(cfgresdir, expname)
                               for (cfgresdir, expname) in cfgresdir_expnames
                               if expname not in frozen_exps]
                      for method, cfgresdir_expnames in method_cfgdirs.items()}

    # method_cfgdirs -> {method1: [(cfgresdir1, expname), ...], ...}
    #exp_meth_dfs = defaultdict(list)
    exp_meth_dfs = {(expname, method): []
                    for method, cfgresdir_expnames in method_cfgdirs.items()
                    for expname in set(x_[1] for x_ in cfgresdir_expnames)}

    for method, cfgresdir_expnames in method_cfgdirs.items():
        my_inds = np.array_split(np.arange(len(cfgresdir_expnames)), mpi_size)[mpi_rank]
        my_cfgresdir_expnames = (cfgresdir_expnames[int(ii)] for ii in my_inds)

        for cfgresdir_tup, expname in my_cfgresdir_expnames:
            proj_resdir_, resdir, cfgdir = cfgresdir_tup
            cfgresdir = f'{proj_resdir_}/{resdir}/{cfgdir}'
            cfgpath = f'{cfgresdir}/config.json'
            prgrspath = f'{cfgresdir}/progress.csv'
            evalpath = f'{cfgresdir}/eval.csv'

            non_existings = [reqfile for reqfile in (cfgpath, prgrspath, evalpath)
                             if not exists(reqfile)]
            for reqfile in non_existings:
                print(f'Warning: {reqfile} does not exist. Skipping directory.')
            if len(non_existings) > 0:
                continue

            with open(cfgpath) as fp:
                cfg_dict = json.load(fp, object_pairs_hook=OrderedDict)

            method = cfg_dict['method']

            # Reading the progress and eval csv files
            try:
                dfp = pd.read_csv(prgrspath)
            except pd.errors.EmptyDataError:
                print(f'Warning: {prgrspath} is empty. Skipping directory.')
                continue

            try:
                dfe = pd.read_csv(evalpath)
            except pd.errors.EmptyDataError:
                print(f'Warning: {evalpath} is empty. Skipping directory.')
                continue

            # Renaming the poorly named / repitive columns
            dfp = dfp.rename(columns=col_renames[method])


            # Sanity Check Assertions
            allowed_keys = xcols[method] + ycols[method] + tcols[method] + ignore_cols + eval_xcols
            over_specified_cols = set(dfp.columns).difference(set(allowed_keys))
            msg_  = f'Extra columns in progress.csv: {over_specified_cols}\n'
            msg_ += f'See {prgrspath} for an actual csv file.'
            msg_ += f'--> add this column name to either method_xcols, \n'
            msg_ += f'    method_ycols, method_tcols, ignore_cols, or \n'
            msg_ += f'    eval_cfgcols in cfg.py'
            assert len(over_specified_cols) == 0, msg_

            allowed_keys = eval_idcols + eval_ycols
            over_specified_cols = set(dfe.columns).difference(set(allowed_keys))
            msg_  = f'Extra columns in eval.csv: {over_specified_cols}\n'
            msg_ += f'See {evalpath} for an actual csv file.'
            msg_ += f'--> add this column name to either eval_idcols, or \n'
            msg_ += f'    eval_ycols in cfg.py'
            assert len(over_specified_cols) == 0, msg_

            # Sorting the evaluation df and setting the 'TimestepsSoFar' column as index
            dfe_proced = dfe.copy(deep=True)
            a = dfe['ckpt_name'].str.split('_', expand=True)
            b = a[1].str.split('.', expand=True)[0].astype(int)
            dfe_proced['TimestepsSoFar'] = b
            x = list({*eval_idcols, 'TimestepsSoFar'}.difference({'eval_seed'}))
            dfe_proced = dfe_proced.groupby(x).mean('eval_seed')
            dfe_proced = dfe_proced.reset_index(level=['TimestepsSoFar'])
            dfe_proced = dfe_proced.sort_values(by='TimestepsSoFar').reset_index(drop=True)

            dfp_proced = dfp.sort_values(by='TimestepsSoFar').reset_index(drop=True)
            # Merging the final df
            df = pd.concat([dfp_proced.set_index('TimestepsSoFar'),
                            dfe_proced.set_index('TimestepsSoFar')], axis=1).reset_index()
            # At this point, df contains the y (i.e., response) columns only.

            # Adding the x (i.e., config) columns
            for col in xcols[method]:
                df[col] = cfg_dict[col]

            # reordering the columns
            dfcols = df.columns.tolist()
            cfgcols = [x for x in dfcols if x in cfg_dict]
            respcols = [x for x in dfcols if x not in cfg_dict]
            df['experiment'] = expname
            # We will introduce the location column and remove it at aggregation
            df['location'] = f'{resdir}/{cfgdir}'
            df = df[['experiment', 'location'] + cfgcols + respcols]

            exp_meth_dfs[(expname, method)].append(df)

    for (expname, method), df_list in exp_meth_dfs.items():
        if len(df_list) == 0:
            continue
        df_summ = pd.concat(df_list, axis=0, ignore_index= True)
        tmp_postfix = tmp_postfix_maker(mpi_rank)
        df_summ.to_csv(f'{tmp_dir}/{expname}_{method}_{tmp_postfix}.csv', index=False)

    #Just waiting until everyone is done!
    if use_mpi:
        comm.Barrier()
    if mpi_rank == 0:
        print('\nEvery rank seems done. Root will start aggregating the csv files...', flush=True)
    time.sleep(1)

    if mpi_rank == 0:
        renewed_outs = set()
        for (expname, method), df_list in exp_meth_dfs.items():
            if expname in frozen_exps:
                print(f'Warning: {expname} is a frozen experiment due to the great reset.')
                print(f'         I will not re-write this experiment summary.')
                continue
            df_cat_list = []
            for rank in range(mpi_size):
                tmp_postfix = tmp_postfix_maker(rank)
                csv_file = f'{tmp_dir}/{expname}_{method}_{tmp_postfix}.csv'
                if not os.path.exists(csv_file):
                    print(f'WARNING: {csv_file} does not exist!', flush=True)
                    print(f'         This could be due to: ', flush=True)
                    print(f'           (1) possibly there was nothing to do for that rank, or', flush=True)
                    print(f'           (2) that particular rank got killed or ran into an issue, or', flush=True)
                    print(f'           (3) the file has not yet been published to the disk. ', flush=True)
                    print(f'         Either way, I will ignore it and move on!', flush=True)
                    continue
                disk_df = pd.read_csv(csv_file)
                df_cat_list.append(disk_df)

            if len(df_cat_list) == 0:
                continue
            agg_df = pd.concat(df_cat_list, axis=0, ignore_index=True)

            all_xtcols = xcols[method] + ['TimestepsSoFar'] #tcols[method]
            all_xtcols = [x for x in all_xtcols if x in agg_df.columns]
            nondup_idxs = ~agg_df.duplicated(subset=all_xtcols, keep=False)
            dplctd_idxs_full = agg_df.duplicated(subset=all_xtcols, keep=False)
            dplctd_idxs_last = agg_df.duplicated(subset=all_xtcols, keep='last')

            dups_df = agg_df
            dups_df = dups_df[all_xtcols + ['location', 'experiment']] # subsetting cols
            dups_df = dups_df[dplctd_idxs_full] # slicing duplicate rows
            dups_df = dups_df.drop('TimestepsSoFar', 1).drop_duplicates()

            agg_df = agg_df[dplctd_idxs_last | nondup_idxs] # only keep the last copy
            #agg_df = agg_df.drop('location', 1) # dropping the location column

            if smry_fmt == 'csv':
                agg_expdir = f'{output_dir}/{expname}'
                if exists(agg_expdir) and (agg_expdir not in renewed_outs):
                    shutil.rmtree(agg_expdir)
                os.makedirs(agg_expdir, exist_ok=True)
                renewed_outs.add(agg_expdir)
                agg_df.to_csv(f'{agg_expdir}/{method}.csv', index=False)
            elif smry_fmt == 'h5':
                agg_out = f'{output_dir}/{expname}.h5'
                h5mode = 'a' if agg_out in renewed_outs else 'w'
                renewed_outs.add(agg_out)
                agg_df.to_hdf(agg_out, key=method, mode=h5mode, index=False)
            else:
                raise ValueError(f'summary format {smry_fmt} not implemented.')

            if dups_df.shape[0] > 0:
                if smry_fmt == 'csv':
                    dupcsvpath = f'{output_dir}/{expname}/{method}_dups_csv'
                elif smry_fmt == 'h5':
                    dupcsvpath = f'{output_dir}/{expname}_{method}_dups_csv'
                else:
                    raise ValueError(f'summary format {smry_fmt} not implemented.')
                print(f'Duplicate rows found and recorded at {dupcsvpath}')
                dups_df.to_csv(dupcsvpath)

        print('Wiping out the temperory directory...', flush=True)
        shutil.rmtree(tmp_dir)

if __name__ == '__main__':
    main(use_mpi=True, be_lazy=True)
