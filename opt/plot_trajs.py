import json
import sys
from os.path import dirname, basename, abspath, exists
import os
cwd=dirname(dirname(abspath(__file__)))
os.chdir(cwd)
sys.path.append(cwd)
from cleg.usr.binding import LegCRoller
from envs.leg import SimInterface, GymEnv
import matplotlib.pyplot as plt
import torch
import numpy as np
import glob
from mpi4py import MPI

class TypicalActor(torch.nn.Module):
    def __init__(self, observation_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(observation_dim, 64, bias=True).double()
        self.fc2 = torch.nn.Linear(64, 64, bias=True).double()
        self.fc3_mu = torch.nn.Linear(64, action_dim, bias=True).double()

    def forward(self, x):
        """
        Takes observation vector x and returns a vector mu and a vector std.
        x: state observation
        mu: mean of action distribution
        std: standard deviation of action distribution
        """
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = self.fc3_mu(x)

        return mu
if __name__ == '__main__':
    traindir = './agents/1-StandingLegLH-v1'
    figdir = './opt/figs'
    theta_hip  = -40 / 57.295
    theta_knee = -100 / 57.295
    pos_slider_init = 0.4
    do_summarize = True

    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.size

    snap_no = None
    snap_dirs = glob.glob(f'{traindir}/snapshot_*')
    for i, snap_dir in enumerate(snap_dirs):
        if (i%mpi_size) != mpi_rank:
            continue
        snapdir_bname = basename(snap_dir)
        print(f'Rank %02d: Working on %s' % (mpi_rank, snapdir_bname), flush=True)
        env_basepath = './envs'
        opts_json_path = f'{env_basepath}/agent_interface_options.json'
        pyxml_file = f'{env_basepath}/leg.xml'
        with open(opts_json_path, 'r') as infile:
            interface_options = json.load(infile)
        interface_options['xml_file'] = pyxml_file
        interface_options['slurm_job_file'] = None

        interface_options['theta_hip_init'] = [theta_hip, theta_hip]
        interface_options['theta_knee_init'] = [theta_knee, theta_knee]
        interface_options['pos_slider_init'] = [pos_slider_init] * 2
        interface_metadata = {}
        sim_interface = SimInterface(options=interface_options,
                                     metadata=interface_metadata)
        plot_env = GymEnv(sim_interface)
        obs_dim = plot_env.observation_dim
        act_dim = plot_env.action_dim

        policy = TypicalActor(observation_dim=obs_dim, action_dim=act_dim)
        agent_path = f'{traindir}/{snapdir_bname}/agent_actor'
        policy.load_state_dict(torch.load(agent_path), strict=False)
        policy_np = lambda s_np: policy(torch.from_numpy(s_np)).detach().numpy()

        sim_interface.options['do_obs_noise'] = False
        plot_env = GymEnv(sim_interface)

        fig = plot_env.simulate_and_plot(policy_np)
        if do_summarize:
            axes = fig.get_axes()
            keep_idxs = [0, 5, 6, 14, 16]
            rm_idxs = sorted(list(set(range(len(axes))) - set(keep_idxs)))
            keep_ax_poses = [ax.get_position() for i, ax in enumerate(axes[:len(keep_idxs)])]
            for ax_idx in sorted(rm_idxs)[::-1]:
                if ax_idx <= (len(axes)-1):
                    axes[ax_idx].remove()
            for ax_idx, pos in zip(keep_idxs, keep_ax_poses):
                ax = axes[ax_idx]
                ax.set_position(pos)
            ax = fig.get_axes()[-1]
            ax.set_xlabel('time (s)')
            ax.xaxis.set_tick_params(which='both', labelbottom=True)

        fig.set_dpi(100)
        fig.set_size_inches(8,48)
        fig.set_tight_layout(dict(h_pad=0.1))

        if not(exists(figdir)):
            os.makedirs(figdir, exist_ok=True)
        fig.savefig(f'{figdir}/{snapdir_bname}.pdf', dpi=100, bbox_inches='tight', pad_inches=0.1)
        fig.savefig(f'{figdir}/{snapdir_bname}.png', dpi=200, bbox_inches='tight', pad_inches=0.1)
