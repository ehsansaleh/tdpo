from mpi4py import MPI

import torch, torch.utils.data
import random, collections, math, time, datetime, os, re, numbers, copy, sys
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from scipy import optimize as sci_optimize
from itertools import cycle
import csv, json
from collections import defaultdict

USE_MPI_DOUBLE = False
def reduce_average(local_tensor):
    """Reduce a PyTorch tensor object by averaging it across all
    processes. The result is returned as a tensor on process 0. All
    other processes get a return value of a zero tensor of the correct
    size.
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    local_array = local_tensor.detach().numpy()
    global_array = np.zeros_like(local_array)
    if USE_MPI_DOUBLE:
        comm.Reduce([local_array,MPI.DOUBLE], [global_array,MPI.DOUBLE], op=MPI.SUM, root=0)
    else:
        comm.Reduce(local_array, global_array, op=MPI.SUM, root=0)
    global_array /= size
    global_tensor = torch.tensor(global_array)
    return global_tensor

def allreduce_average(local_tensor):
    """All-reduce a PyTorch tensor object by averaging it across all
    processes. The result is returned as a tensor on all processes.
    """
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    local_array = local_tensor.detach().numpy()
    global_array = np.zeros_like(local_array)
    if USE_MPI_DOUBLE:
        comm.Allreduce([local_array,MPI.DOUBLE], [global_array,MPI.DOUBLE], op=MPI.SUM)
    else:
        comm.Allreduce(local_array, global_array, op=MPI.SUM)
    global_array /= size
    global_tensor = torch.tensor(global_array)
    return global_tensor

def allreduce_average_tuple(local_tuple):
    local_flat = torch._utils._flatten_dense_tensors(local_tuple)
    global_flat = allreduce_average(local_flat)
    global_tuple = torch._utils._unflatten_dense_tensors(global_flat, local_tuple)
    return global_tuple

def reduce_average_tuple(local_tuple):
    local_flat = torch._utils._flatten_dense_tensors(local_tuple)
    global_flat = reduce_average(local_flat)
    global_tuple = torch._utils._unflatten_dense_tensors(global_flat, local_tuple)
    return global_tuple

def broadcast_ds(d):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if isinstance(d, dict):
        generator_ = sorted(d.keys())
    elif isinstance(d, (tuple,list)):
        generator_ = range(len(d))
    else:
        raise Exception(f'Unknown type ds: {type(d)}')

    if rank == 0:
        for k in generator_:
            comm.Bcast(d[k].numpy(), root=0)
    else:
        for k in generator_:
            param = np.empty_like(d[k].numpy())
            comm.Bcast(param, root=0)
            param_t = torch.tensor(param).double()
            d[k].copy_(param_t)
            del param_t, param
    return d

def gather_rows(local_tensor):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_array = local_tensor.detach().numpy()
    assert len(local_array.shape) == 2
    num_rows = local_array.shape[0]
    num_cols = local_array.shape[1]
    global_array = None
    if rank == 0:
        global_array = np.empty((num_rows * size, num_cols), dtype=local_array.dtype)
    comm.Gather(local_array, global_array, root=0)
    if rank == 0:
        global_tensor = torch.tensor(global_array)
        return global_tensor
    else:
        return None

def gather_np(local_array):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_rows = local_array.shape[0]
    num_cols = local_array.shape[1]
    global_array = None
    if rank == 0:
        global_array = np.empty((num_rows * size, num_cols), dtype=local_array.dtype)
    comm.Gather(local_array, global_array, root=0)
    return global_array

def repeat_and_divide(np_var, grouping, repeats, mpi_size=None):
    # This function groups the first axis, and puts every `grouping` rows in one bucket
    # Then it repeats the buckets `repeats` number of times (in an interleaved manner)
    # Then recovers the old shape, with only the first dimension changed.
    # Then it inserts a dummy dimension mpi_size at `axis=0`.
    # This is essentially done to faciliatate dividing the first dimension among the MPI workers by a scatter operation.

    # Ex:
    #   >>> a = np.arange(12).reshape(4,3)
    #   >>> b = repeat_and_divide(np_var=a, grouping=1, repeats=2, mpi_size=8)
    #   >>> b.shape
    #       --> (8,1,3)
    #   >>> np.array_equal(b[0], b[1])  # This is because the first two workers should get the same stuff.
    #       --> True
    if np_var.ndim < 2:
        np_var = np_var.reshape(-1, 1)
    shape_postfix = np_var.shape[1:]
    np_var_expanded = np.expand_dims(np_var, axis=0).reshape(-1, grouping, *shape_postfix)
    np_var_tiled = np.repeat(np_var_expanded, repeats=repeats, axis=0).reshape(-1, *shape_postfix)
    if mpi_size is not None:
        np_var_tiled = np_var_tiled.reshape(mpi_size, -1, *shape_postfix)
    return np_var_tiled

def prepare_for_scatter(np_var, d2, mpi_size, **kwargs):
    assert mpi_size == (d2 * np_var.shape[0] // (d2-1)) # Because of the way we've compiled the vine data
    return repeat_and_divide(np_var=np_var, grouping=d2-1, repeats=d2, mpi_size=mpi_size, **kwargs)

def a2s(a):
    return np.array2string(a, formatter={'float_kind':lambda x: f'{x:.2f}'})

def numpy_to_matlab(A, ndigits=2, wtype='f'):
    """numpy_to_matlab(A, ndigits=2, wtype='f')

    This function assumes that A is one of these things:

        - a number (float or complex)
        - a 2D ndarray (float or complex)

    It returns A as a MATLAB-formatted string in which each number has "ndigits"
    digits after the decimal and is formatted as "wtype" (e.g., 'f', 'g', etc.).
    """
    if np.isscalar(A):
        A_str = '{:.{indigits}{iwtype}}'.format(A, indigits=ndigits, iwtype=wtype)
        return A_str
    else:
        s = A.shape
        m = s[0]
        n = s[1]
        A_str = '['
        for i in range(0, m):
            for j in range(0, n):
                A_str += '{:.{indigits}{iwtype}}'.format(A[i, j], indigits=ndigits, iwtype=wtype)
                if j == n - 1:
                    if i == m - 1:
                        A_str += ']'
                    else:
                        A_str += '; '
                else:
                    A_str += ' '
        return A_str

def fc_to_matlab(fc, act='tanh', name='fc', ndigits=8, wtype='f'):
    """
    Assumes fc is a fully-connected NN layer of type torch.nn.Linear with
    a given activation function applied to the output.

    (If act=None, no activation function is applied.)

    Returns a string with a MATLAB function definition that is equivalent to
    this layer. The function has the name 'name'.
    """
    A = numpy_to_matlab(fc.weight.data.numpy(), ndigits=ndigits, wtype=wtype)
    b = numpy_to_matlab(np.reshape(fc.bias.data.numpy(), (-1, 1)), ndigits=ndigits, wtype=wtype)
    if act is None:
        sout = f'y = A * x + b;'
    else:
        sout = f'y = {act}(A * x + b);'
    s = f'function y = {name}(x)\n' + \
        f'A = {A};\n' + \
        f'b = {b};\n' + \
        f'{sout}\n' + \
        f'end\n'
    return s

def prep_lsc(line_search_coeffs, mpi_size):
    # Next, we create the array `lsc_full` out of `line_search_coeffs`,
    # and append repetitive elements until its size becomes a multiple of mpi size.
    lsc_num = line_search_coeffs.size
    lsc_unq = np.unique(line_search_coeffs)
    lsc_unq.sort()
    remainder = (-lsc_num) % mpi_size
    nelem = (lsc_num + remainder)
    assert nelem % mpi_size == 0, f'nelem = {nelem}, mpi_size = {mpi_size}'
    ncols = mpi_size
    nrows = nelem // mpi_size
    a = int(np.ceil(nelem / lsc_num))
    lsc_full = np.array(line_search_coeffs.tolist() * a)
    assert len(lsc_full) >= nelem, f'len(lsc_full) = {len(lsc_full)}, nelem={nelem}'
    lsc_full = lsc_full[:nelem]
    lsc_full.sort()

    # Next, we determine the proper seed for each of the `lsc_full` elements.
    # We make sure that the 1st occurance of each coefficient gets 0
    # We make sure that the 2nd occurance of each coefficient gets `1000`
    # We make sure that the 3rd occurance of each coefficient gets `2000`
    # etc.
    ls_seeds = []
    lsc_unq_cnts = {c:0 for c in lsc_unq}
    for _, c in enumerate(lsc_full):
        ls_seeds.append(lsc_unq_cnts[c] * 1000)
        lsc_unq_cnts[c] += 1
    ls_seeds_full = np.array(ls_seeds)
    return lsc_full, ls_seeds_full

class Actor(torch.nn.Module):
    def __init__(self, observation_dim, action_dim, **kwargs):
        super(Actor, self).__init__()
        self.hidden_units = kwargs.get('actor_hidden_units', [64, 64])
        self.output_scale = kwargs.get('output_scale', 1.)
        self.output_activation = kwargs.get('output_activation', None)
        self.use_bias = kwargs.get('use_bias', True)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        if len(self.hidden_units) == 3:
            h1, h2, h3 = tuple(self.hidden_units)
            self.fc1 = torch.nn.Linear(observation_dim, h1, bias=self.use_bias).double()
            self.fc2 = torch.nn.Linear(h1, h2, bias=self.use_bias).double()
            self.fc3 = torch.nn.Linear(h2, h3, bias=self.use_bias).double()
            self.fc4_mu = torch.nn.Linear(h3, action_dim, bias=self.use_bias).double()
        elif len(self.hidden_units) == 2:
            h1, h2 = tuple(self.hidden_units)
            self.fc1 = torch.nn.Linear(observation_dim, h1, bias=self.use_bias).double()
            self.fc2 = torch.nn.Linear(h1, h2, bias=self.use_bias).double()
            self.fc3_mu = torch.nn.Linear(h2, action_dim, bias=self.use_bias).double()
        elif len(self.hidden_units) == 1:
            h1 = tuple(self.hidden_units)[0]
            self.fc1 = torch.nn.Linear(observation_dim, h1, bias=self.use_bias).double()
            self.fc2_mu = torch.nn.Linear(h1, action_dim, bias=self.use_bias).double()
        elif len(self.hidden_units) == 0:
            self.fc1_mu = torch.nn.Linear(observation_dim, action_dim, bias=self.use_bias).double()
        else:
            raise Exception(f'Unsupported Architecture: {self.hidden_units}')
        # You must cast the zero tensor as double() and not the Parameter - if
        # you try to cast the Parameter, it ceases to be a Parameter and simply
        # becomes a zero tensor again.
        self.fixed_std = kwargs.get('actor_fixed_std', None)
        if self.fixed_std is None:
            self.param_std = torch.nn.Parameter(-0.5 * torch.ones(action_dim).double())
        else:
            self.fixed_std = torch.ones(action_dim).double() * self.fixed_std  # It should not be a parameter
            self.fixed_std.detach_()
        # self.param_std = torch.nn.Parameter(torch.zeros(action_dim).double())

    def forward(self, x):
        """
        Takes observation vector x and returns a vector mu and a vector std.
        x: state observation
        mu: mean of action distribution
        std: standard deviation of action distribution
        """

        col_reps = 1
        if x.dim() == 2:
            if x.shape[1] % self.observation_dim == 0:
                col_reps = int(x.shape[1]/self.observation_dim)
                x = x.reshape(x.shape[0] * col_reps, self.observation_dim)
            else:
                raise Exception('Unsupported input dimensions', x.shape)
        elif x.dim() > 2:
            raise Exception('Too many dimensions', x.shape)

        if len(self.hidden_units) == 0:
            mu = self.fc1_mu(x)
        elif len(self.hidden_units) == 1:
            x = torch.tanh(self.fc1(x))
            mu = self.fc2_mu(x)
        elif len(self.hidden_units) == 2:
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            mu = self.fc3_mu(x)
        elif len(self.hidden_units) == 3:
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            mu = self.fc4_mu(x)
        else:
            raise Exception(f'Unsupported Architecture: {self.hidden_units}')

        if self.fixed_std is not None:
            std = self.fixed_std
        else:
            std = torch.exp(self.param_std)
            # std = torch.sigmoid(self.param_std)    # adding a small number may improve robustness

        if self.output_activation is not None:
            mu = self.output_activation(mu)
            #assert torch.sum(mu > 1) < 1

        mu, std = (mu * self.output_scale, std * self.output_scale)

        if col_reps > 1:
            mu = mu.reshape(-1, self.action_dim * col_reps)
            std = std.repeat(col_reps)

        return mu, std

    def reset(self):
        pass

    def reset_std(self, value=0.6):
        """
        Resets the std to its initial value.
        """
        if self.fixed_std is None:
            self.param_std.data = np.log(value) * torch.ones(self.param_std.shape[0]).double()
        else:
            self.fixed_std = torch.ones(self.action_dim).double() * value
            self.fixed_std.detach_()

    def reset_mu(self, mu):
        """
        In output layer, sets weight to zero and bias to mu, which must be an
        ndarray of the right size. The output of the network will then be mu.
        """
        for name, p in self.fc3_mu.named_parameters():
            if 'weight' in name:
                p.data = torch.zeros_like(p.data).double()
            elif 'bias' in name:
                p.data = torch.tensor(mu).double()

    def broadcast(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            d = self.state_dict()
            for k in sorted(d.keys()):
                comm.Bcast(d[k].numpy(), root=0)
        else:
            d = self.state_dict()
            for k in sorted(d.keys()):
                param = np.empty_like(d[k].numpy())
                comm.Bcast(param, root=0)
                d[k] = torch.tensor(param).double()
            self.load_state_dict(d, strict=True)

    def grad_reduce_average(self):
        for param in self.parameters():
            global_data = reduce_average(param.grad.data)
            param.grad.data = global_data

    def grad_allreduce_average(self):
        for param in self.parameters():
            global_data = allreduce_average(param.grad.data)
            param.grad.data = global_data

class RBFActor(torch.nn.Module):
    def __init__(self, observation_dim, action_dim, **kwargs):
        super(RBFActor, self).__init__()
        self.rbf_units = kwargs.get('rbf_units', 200)
        self.output_scale = kwargs.get('output_scale', 1.)
        self.output_activation = kwargs.get('output_activation', None)
        self.use_bias = kwargs.get('use_bias', False)

        self.initial_rbf_std = kwargs.get('initial_rbf_std', 1.)
        self.use_independent_rbf_std_per_unit = kwargs.get('use_independent_rbf_std_per_unit', False)
        self.fixed_inner_rbf_std = kwargs.get('fixed_inner_rbf_std', True)

        self.observation_dim = observation_dim
        self.action_dim = action_dim


        # Parameter Construction:
        #   1) rbf_means
        self.rbf_means = torch.nn.Parameter(torch.randn(self.rbf_units, self.observation_dim).double())

        #   2) log_rbf_stds
        if self.fixed_inner_rbf_std:
            rbf_stds = torch.tensor(self.initial_rbf_std).double() * torch.ones(self.observation_dim).double()
            rbf_stds = rbf_stds.reshape(-1, self.observation_dim)
            if not self.use_independent_rbf_std_per_unit:
                assert rbf_stds.shape[0] == 1, 'If use_independent_rbf_std_per_unit==False, rbf_stds should be a single row.'
            self.log_rbf_stds = torch.log(rbf_stds)
            self.log_rbf_stds.detach_()
        else:
            rbf_stds = torch.tensor(self.initial_rbf_std).double() * torch.ones(self.observation_dim).double()
            rbf_stds = rbf_stds.reshape(-1, self.observation_dim)
            if self.use_independent_rbf_std_per_unit:
                rbf_stds = rbf_stds * torch.ones(self.rbf_units, 1).double()
            else:
                assert rbf_stds.shape[0] == 1, 'If use_independent_rbf_std_per_unit==False, rbf_stds should be a single row.'
            self.log_rbf_stds = torch.nn.Parameter(torch.log(rbf_stds))

        #   3) Linear Layer
        self.fc_mu = torch.nn.Linear(self.rbf_units, action_dim, bias=self.use_bias).double()

        #   4) Action std
        self.fixed_std = kwargs.get('actor_fixed_std', None)
        if self.fixed_std is None:
            self.param_std = torch.nn.Parameter(-0.5 * torch.ones(action_dim).double())
        else:
            self.fixed_std = torch.ones(action_dim).double() * self.fixed_std  # It should not be a parameter
            self.fixed_std.detach_()

    def forward(self, x):
        """
        Takes observation vector x and returns a vector mu and a vector std.
        x: state observation
        mu: mean of action distribution
        std: standard deviation of action distribution
        """

        col_reps = 1
        was_x_1d = False
        if x.dim() == 1:
            was_x_1d = True
            x = x.reshape(1, self.observation_dim)
        elif x.dim() == 2:
            if x.shape[1] % self.observation_dim == 0:
                col_reps = int(x.shape[1]/self.observation_dim)
                x = x.reshape(x.shape[0] * col_reps, self.observation_dim)
            else:
                raise Exception('Unsupported input dimensions', x.shape)
        elif x.dim() > 2:
            raise Exception('Too many dimensions', x.shape)


        assert x.shape[1] == self.rbf_means.shape[1] == self.observation_dim
        N, M = x.shape[0], self.rbf_means.shape[0]
        if self.use_independent_rbf_std_per_unit:
            rbf_means_unsqueezed = torch.unsqueeze(self.rbf_means, 0) # shape=(1,M,d)
            x_unsqueezed = torch.unsqueeze(x, 1) # shape=(N,1,d)
            rbf_stds_unsquezed = torch.unsqueeze(torch.exp(self.log_rbf_stds), 0) # shape=(1,M,d)
            x_rbf_mean_diff = x_unsqueezed - rbf_means_unsqueezed # shape=(N,M,d)
            x_rbf_mean_diff_standardized = (x_rbf_mean_diff/rbf_stds_unsquezed) # shape=(N,M,d)
            sq_dist_mat = (x_rbf_mean_diff_standardized ** 2).sum(2) # shape=(N,M)
        else:
            rbf_stds = torch.exp(self.log_rbf_stds)
            y_standardized = self.rbf_means / rbf_stds
            x_standardized = x / rbf_stds
            x_sq = (x_standardized**2).sum(1).view(-1, 1)
            y_sq = (y_standardized**2).sum(1).view(1, -1)
            sq_dist_mat = x_sq + y_sq - 2.0 * torch.mm(x_standardized, torch.transpose(y_standardized, 0, 1))
        assert sq_dist_mat.shape == (N, M)
        # I won't use the gaussian normalization. Maybe it's a bad decision, but it made sense at the time I wrote this.
        gauss_kern = torch.exp(-sq_dist_mat / 2.)
        mu = self.fc_mu(gauss_kern)

        if self.fixed_std is not None:
            std = self.fixed_std
        else:
            std = torch.exp(self.param_std)
            # std = torch.sigmoid(self.param_std)    # adding a small number may improve robustness

        if self.output_activation is not None:
            mu = self.output_activation(mu)
            #assert torch.sum(mu > 1) < 1

        mu, std = (mu * self.output_scale, std * self.output_scale)

        if col_reps > 1:
            mu = mu.reshape(-1, self.action_dim * col_reps)
            std = std.repeat(col_reps)

        if was_x_1d:
            mu = mu.reshape(-1)
            std = std.reshape(-1)

        return mu, std

    def reset(self):
        pass

    def reset_std(self, value=0.6):
        """
        Resets the std to its initial value.
        """
        if self.fixed_std is None:
            self.param_std.data = np.log(value) * torch.ones(self.param_std.shape[0]).double()
        else:
            self.fixed_std = torch.ones(self.action_dim).double() * value
            self.fixed_std.detach_()

    def reset_mu(self, mu):
        """
        In output layer, sets weight to zero and bias to mu, which must be an
        ndarray of the right size. The output of the network will then be mu.
        """
        for name, p in self.fc_mu.named_parameters():
            if 'weight' in name:
                p.data = torch.zeros_like(p.data).double()
            elif 'bias' in name:
                p.data = torch.tensor(mu).double()

    def broadcast(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            d = self.state_dict()
            for k in sorted(d.keys()):
                comm.Bcast(d[k].numpy(), root=0)
        else:
            d = self.state_dict()
            for k in sorted(d.keys()):
                param = np.empty_like(d[k].numpy())
                comm.Bcast(param, root=0)
                d[k] = torch.tensor(param).double()
            self.load_state_dict(d, strict=True)

    def grad_reduce_average(self):
        for param in self.parameters():
            global_data = reduce_average(param.grad.data)
            param.grad.data = global_data

    def grad_allreduce_average(self):
        for param in self.parameters():
            global_data = allreduce_average(param.grad.data)
            param.grad.data = global_data

class Critic(torch.nn.Module):
    def __init__(self, observation_dim, action_dim, **kwargs):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(observation_dim, 64).double()
        self.fc2 = torch.nn.Linear(64, 64).double()
        self.fc3_V = torch.nn.Linear(64, 1).double()

    def forward(self, x):
        """
        Takes observation vector x and returns a scalar V.
        V: scalar value function
        """
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        V = self.fc3_V(x)
        return V

    def broadcast(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            d = self.state_dict()
            for k in sorted(d.keys()):
                comm.Bcast(d[k].numpy(), root=0)
        else:
            d = self.state_dict()
            for k in sorted(d.keys()):
                param = np.empty_like(d[k].numpy())
                comm.Bcast(param, root=0)
                d[k] = torch.tensor(param).double()
            self.load_state_dict(d, strict=True)


    def grad_reduce_average(self):
        for param in self.parameters():
            global_data = reduce_average(param.grad.data)
            param.grad.data = global_data


class AClosure:
    def __init__(self, policy_params, identity_stabilization=None, mpi_enabled=True,
                 GSq2WSq_ratio=0, obs2act_jac='exact', average_over_rows=False):
        self.policy_params = policy_params
        self.policy = None
        self.action_dist_params = None
        self.identity_stabilization = identity_stabilization
        self.orig_identity_stabilization = identity_stabilization
        self.mpi_enabled = mpi_enabled
        self.GSq2WSq_ratio = GSq2WSq_ratio
        self.obs2act_jac = obs2act_jac
        self.average_over_rows = average_over_rows
        self.num_rows = None
        self.sample_coeffs = None
        assert self.obs2act_jac in ('finite_diff', 'exact')

    def set_policy(self, policy):
        self.policy = policy

    def generate_intermediate_params(self, states_tensor):
        assert len(states_tensor.shape) == 2
        assert self.policy is not None
        #if self.GSq2WSq_ratio > 0 and self.obs2act_jac == 'finite_diff':
            #assert states_tensor.requires_grad

        mu_std = self.policy(states_tensor)
        if not isinstance(mu_std, (tuple, list)):
            mu_std = (mu_std,)

        obs_dim = states_tensor.shape[1]

        out_dict = {'action_dist_params':mu_std}
        if self.GSq2WSq_ratio > 0:
            eps = 1e-6
            if self.obs2act_jac == 'finite_diff':
                jac_pairs = []
                for obs_dim_idx in range(obs_dim):
                    perturbed_state = states_tensor.clone().detach()
                    perturbed_state[:, obs_dim_idx] += eps
                    perturbed_mu_std = self.policy(perturbed_state)
                    jac_pair = tuple((pert_x_-x_)/eps for pert_x_, x_ in zip(perturbed_mu_std, mu_std))
                    jac_pairs.append(jac_pair)

                mu_std_jacobians = []
                for i in range(len(jac_pairs[0])):
                    jac_cat = torch.cat([jac_pair[i] for jac_pair in jac_pairs], dim=1 if len(jac_pairs[0][i].shape)>=2 else 0)
                    mu_std_jacobians.append(jac_cat)
                mu_std_jacobians = tuple(mu_std_jacobians)

                out_dict['action_jaccobians'] = mu_std_jacobians

            elif self.obs2act_jac == 'exact':
                def actions_reduce_row(s_):
                    mu_std_jacs = []
                    for a_ in self.policy(s_):
                        # for std: std is usually 1-dim
                        if len(a_.shape) == 1:
                            mu_std_jacs.append(a_)
                        else:
                            mu_std_jacs.append(a_.sum(dim=0))
                    return tuple(mu_std_jacs)

                mu_std_jacobians = torch.autograd.functional.jacobian(actions_reduce_row, states_tensor, create_graph=True, strict=False)

                out_dict['action_jaccobians'] = mu_std_jacobians

        return out_dict

    def set_action_dist_params(self, states_tensor):
        self.num_rows = states_tensor.shape[0]
        if self.mpi_enabled:
            # Some of the workers may have smaller samples than traj length,
            # since some environments may cut off the trajectory in the middle.
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            nom_row_avg_rank = comm.allreduce(self.num_rows, op=MPI.SUM)
            self.num_rows = float(nom_row_avg_rank) / size
        new_action_dist_params = self.generate_intermediate_params(states_tensor)
        new_action_dist_params = {key: tuple(a_ for a_ in val if a_.requires_grad) for key,val in new_action_dist_params.items()}
        if self.action_dist_params is None:
            self.action_dist_params = new_action_dist_params
            self.t = self.create_t(new_action_dist_params)
            # self.t is a dict like {'action_dist_params':(tensor1, tensor2,),
            #                        'action_jaccobians': (tensor3, tensor4,)}
        else:
            self.action_dist_params = new_action_dist_params

            same_old_shape = True
            for key in new_action_dist_params.keys():
                same_old_shape = same_old_shape and np.all([x_.shape == y_.shape for x_, y_ in zip(self.t[key], new_action_dist_params[key])])
            if not same_old_shape:
                for key, val in self.t.items():
                    for x_ in val:
                        del x_
                self.t = self.create_t(new_action_dist_params)

    def create_t(self, new_action_dist_params):
        t_dict = {}
        for key, val in new_action_dist_params.items():
            t_list = []
            for x_ in val:
                a_ = torch.ones_like(x_).double()
                if x_.requires_grad:
                    a_ = torch.nn.Parameter(a_)
                t_list.append(a_)
            t_dict[key] = tuple(t_list)
        return t_dict

    def set_w2_weights(self, sample_coeffs):
        assert torch.is_tensor(sample_coeffs)
        assert sample_coeffs.ndim == 1
        act_dim = self.action_dist_params['action_dist_params'][0].shape[1]
        self.sample_coeffs = sample_coeffs.detach()
        self.sample_coeffs_repeated = self.sample_coeffs.reshape(-1, 1).repeat(1, act_dim).detach()
        self.sample_coeffs_sum = torch.sum(self.sample_coeffs).detach()

        self.identity_stabilization = self.orig_identity_stabilization * self.sample_coeffs_sum.item()

    def w2_M_closure(self, Jv_seq):
        if isinstance(Jv_seq, tuple):
            if len(Jv_seq) == 2:
                J_mu, J_std = Jv_seq

                assert isinstance(J_mu, torch.Tensor)
                assert isinstance(J_std, torch.Tensor)

                num_samples = J_mu.numel() / J_std.numel()
                assert num_samples == self.num_rows or self.mpi_enabled
                if self.sample_coeffs is None:
                    coeff1, coeff2 = 1, 1
                else:
                    coeff1, coeff2 = self.sample_coeffs_repeated.reshape(*J_mu.shape).detach(), self.sample_coeffs_sum.detach()

                if self.average_over_rows:
                    return (J_mu * coeff1 / self.num_rows, J_std * coeff2)
                else:
                    return (J_mu * coeff1, J_std * num_samples * coeff2)

            elif len(Jv_seq) == 1:
                return (self.w2_M_closure(Jv_seq[0]),)

        elif isinstance(Jv_seq, torch.Tensor):
            if self.sample_coeffs is None:
                coeff = 1
            else:
                coeff = self.sample_coeffs_repeated.reshape(*Jv_seq.shape)

            if self.average_over_rows:
                return Jv_seq * coeff / self.num_rows
            else:
                return Jv_seq * coeff
        else:
            raise Exception(f'Unknown type of Jv_seq: {type(Jv_seq)}    {Jv_seq}')

    def g2_M_closure(self, Jv_seq):
        return self.w2_M_closure(Jv_seq)

    def consolidate_nones(self, nablas, params):
        consolidated_nablas = []
        for p_der_, p_ in zip(nablas, params):
            if p_der_ is None:
                consolidated_nablas.append(torch.zeros_like(p_))
            else:
                consolidated_nablas.append(p_der_)
        return tuple(consolidated_nablas)

    def lite_vec_prod(self, vec, t, action_dist_params, M_closure, identity_stabilization=None,
                      flat_out=True, allow_unused=False, **kwargs):
        from torch.autograd import Variable
        msg_ = 'action_dist_params should be set using self.set_action_dist_params method prior to this call.'
        assert action_dist_params is not None, msg_

        # Step 1
        Jt = torch.stack(tuple(torch.sum(x * y) for x, y in zip(t, action_dist_params))).sum()
        # Step 2
        nabla_theta_Jt_seq = torch.autograd.grad(Jt, self.policy_params, grad_outputs=None, create_graph=True,
                                                 only_inputs=True, allow_unused=allow_unused)
        if allow_unused:
            nabla_theta_Jt_seq = self.consolidate_nones(nabla_theta_Jt_seq, self.policy_params)

        # Step 3
        if isinstance(vec, tuple):
            # In case the input is a sequence of tensors
            nabla_theta_Jt_v = torch.stack(
                tuple(torch.sum(x * Variable(y)) for x, y in zip(nabla_theta_Jt_seq, vec))).sum()
        else:
            # In case the input is a flattened tensors
            nabla_theta_Jt = torch._utils._flatten_dense_tensors(nabla_theta_Jt_seq)
            nabla_theta_Jt_v = torch.sum(nabla_theta_Jt * Variable(vec))

        # Step 4
        Jv = torch.autograd.grad(nabla_theta_Jt_v, t, grad_outputs=None,
                                 create_graph=False, retain_graph=True,
                                 only_inputs=True, allow_unused=allow_unused)
        if allow_unused:
            Jv = self.consolidate_nones(Jv, t)
        # Step 5
        MJv = M_closure(Jv)

        # Step 6
        MJv_prod = torch.stack(tuple(torch.sum(x * y) for x, y in zip(MJv, action_dist_params))).sum()

        JtMJV = torch.autograd.grad(MJv_prod, self.policy_params, grad_outputs=None, create_graph=True,
                                    only_inputs=True, allow_unused=allow_unused)
        if allow_unused:
            JtMJV = self.consolidate_nones(JtMJV, self.policy_params)

        if identity_stabilization is not None:
            if isinstance(vec, tuple):
                for x_, y_ in zip(JtMJV, vec):
                    x_.data.add_(identity_stabilization, y_)
            else:
                offset = 0
                for x_ in JtMJV:
                    numel = x_.numel()
                    y_ = vec.narrow(0, offset, numel).view_as(x_)
                    x_.data.add_(identity_stabilization, y_)
                    offset += numel

        if flat_out:
            JtMJV = torch._utils._flatten_dense_tensors(JtMJV)
            if self.mpi_enabled:
                JtMJV = allreduce_average(JtMJV)
        else:
            if self.mpi_enabled:
                allreduced_JtMJV_ = tuple(allreduce_average(param.data) for param in JtMJV)
            else:
                allreduced_JtMJV_ = tuple((param.data) for param in JtMJV)
            JtMJV = allreduced_JtMJV_

        return JtMJV

    def hessian_vec_prod(self, vec, t, action_dist_params, M_closure, identity_stabilization=None, flat_out=True, allow_unused=False, **kwargs):
        raise Exception('Not implemented')

    def __call__(self, vec, flat_out=True, GSq2WSq_ratio=None, **kwargs):
        if GSq2WSq_ratio is None:
            GSq2WSq_ratio =  self.GSq2WSq_ratio
        w2vp = self.lite_vec_prod(vec, t=self.t['action_dist_params'],
                                  action_dist_params = self.action_dist_params['action_dist_params'],
                                  M_closure = self.w2_M_closure,
                                  identity_stabilization = self.identity_stabilization,
                                  flat_out = flat_out,
                                  allow_unused = False,
                                  **kwargs)

        if GSq2WSq_ratio > 0:
            g2vp = self.lite_vec_prod(vec, t=self.t['action_jaccobians'],
                                      action_dist_params = self.action_dist_params['action_jaccobians'],
                                      M_closure = self.g2_M_closure,
                                      identity_stabilization = None,
                                      flat_out = flat_out,
                                      allow_unused = (self.obs2act_jac == 'exact'),
                                      **kwargs)

        if GSq2WSq_ratio > 0:
            if flat_out:
                vp = w2vp + GSq2WSq_ratio * g2vp
            else:
                assert isinstance(w2vp, (tuple))
                vp = tuple(w_ + GSq2WSq_ratio * g_ for (w_, g_) in zip(w2vp, g2vp))
        else:
            vp = w2vp

        return vp

class Vine_Policy:
    def __init__(self, grp_rank_, env_ac_dim, options, actor, d2=2,
                 np_random=np.random, exp_noise_generator=None, exp_act_post_proc=None):
        self.d1 = None
        self.d2 = d2
        self.grp_rank_ = grp_rank_
        self.np_random = np_random
        self.exp_noise_generator = exp_noise_generator
        self.env_ac_dim = env_ac_dim
        self.actor = actor
        self.options = options

        self.a_disturbance = self.options['vine_params'].get('action_disturbance', 0.01)

        assert self.d2 >= 2
        assert 0 <= self.grp_rank_ < self.d2

        self.act_dim_iterator_ = None
        if self.exp_noise_generator is None:
            self.act_dim_iterator_ = cycle(range(self.env_ac_dim))
        self.exp_act_post_proc = exp_act_post_proc or (lambda noised_a_, greedy_a_: noised_a_)

    def set_d1(self, d1):
        self.d1 = d1

    def reset(self):
        assert self.d1 is not None, 'use set_d1 before resetting this object'
        if self.exp_noise_generator is None:
            self.rand_seq = np.zeros((self.d2-1, self.d1, self.env_ac_dim))
            for i_ in range(self.d2-1):
                curr_dim = next(self.act_dim_iterator_)
                cst_d1_noise = self.np_random.randn() * self.a_disturbance
                for j_ in range(self.d1):
                    self.rand_seq[i_][j_][curr_dim] = cst_d1_noise
            self.rand_seq = self.rand_seq.reshape((self.d2-1) * self.d1, self.env_ac_dim)
        else:
            self.rand_seq = self.exp_noise_generator((self.d2-1) * self.d1, self.env_ac_dim, self.np_random)
            self.rand_seq = self.rand_seq.reshape((self.d2-1) * self.d1, self.env_ac_dim)
        return self.rand_seq

    def __call__(self, state_tensor, t_now, t_init_exp):
        with torch.no_grad():
            greedy_a_, std_ = self.actor(state_tensor)
            assert len(list(greedy_a_.shape)) == 1
            if t_init_exp <= t_now < (t_init_exp + self.grp_rank_ * self.d1):
                noised_a_ = greedy_a_ + torch.from_numpy(self.rand_seq[t_now - t_init_exp].reshape(greedy_a_.shape)).double()
                a_ = self.exp_act_post_proc(noised_a_, greedy_a_)
            else:
                a_ = greedy_a_
            return a_, std_ * 0.

class GenericPolicy:
    def __init__(self, vine_exp_policy, is_greedy=True):
        self.t = 0
        self.vine_exp_policy = vine_exp_policy
        self.is_greedy = is_greedy

    def __call__(self, inp_):
        with torch.no_grad():
            self.t = self.t + 1
            if self.is_greedy:
                t_exp_init = -np.inf
            else:
                assert False, 'Not implemented yet. Need random exploration time-step.'
                t_exp_init = 0
            (mu, std) = self.vine_exp_policy(torch.from_numpy(inp_).double(), self.t, t_exp_init)
            return mu.numpy()

class Explorer:
    def __init__(self, init_mu, init_std, lr=1e-4, max_buff_len=1, epochs=1):
        self.use_mu = (init_mu is not None)
        assert (init_std is not None)

        self.params = dict()

        init_std_np = np.array(init_std).reshape(-1)
        init_logstd_np = np.log(init_std_np)
        self.log_std = torch.nn.Parameter(torch.from_numpy(init_logstd_np).double())
        self.params['log_std'] = self.log_std

        act_dim = init_std_np.size
        if self.use_mu:
            msg_ = f'init_std.shape = {init_std_np.shape}, init_mu.shape={init_mu_np.shape}'
            assert init_std_np.size == act_dim, msg_

            init_mu_np = np.array(init_mu).reshape(-1)
            self.mu = torch.nn.Parameter(torch.from_numpy(init_mu_np).double())
            self.params['mu'] = self.mu
        else:
            self.mu = torch.zeros(act_dim, dtype=torch.double)

        self.optimizer = torch.optim.Adam(list(self.params.values()), lr=lr)

        self.max_buff_len = max_buff_len
        self.epochs = epochs
        self.data_buffer = []

    @property
    def mu_np(self):
        return self.mu.detach().cpu().numpy()

    @property
    def log_std_np(self):
        return self.log_std.detach().cpu().numpy()

    @property
    def std_np(self):
        return np.exp(self.log_std.detach().cpu().numpy())

    def __call__(self, rows, cols, np_random, depth=None):
        """
        This function samples exploratory noises for a number of
          1) time-steps, and
          2) action-dimensions.

        Arguments:
          rows (int): The `rows` argument will represent the "time" progression.
                      That is, each row will correspond to a single time-step.
                      The value `(self.d2-1) * self.d1` is usually passed to this
                      argument in the xpo code.

          cols (int): The `cols` argument will be the number of action dimensions.
                      That is, each column will correspond to one of the action
                      dimensions. The value `self.env_ac_dim` is usually passed to
                      this argument in the xpo code.

          np_random (np.random.RandomState): The numpy random number generator.

          depth (int): This will be used generate a batch of exploratory
                       noises to be used for multiple rollouts.

        Output:
          out (np.array): A numpy array with the shape (rows, cols)
        """
        act_dim = cols
        time_steps = rows
        if self.use_mu:
            msg_ = f'self.mu_np.size = {self.mu_np.size}, cols = {cols}'
            assert self.mu_np.size == act_dim, msg_
        msg_ = f'self.log_std_np.size = {self.log_std_np.size}, cols = {cols}'
        assert self.log_std_np.size == act_dim, msg_

        if depth is None:
            depth_ = 1
        else:
            depth_ = depth

        # Step 1) We sample a bunch of normal random variables for a single time-step.
        normal_vars = np_random.randn(depth_, 1, act_dim)
        # Step 2) We scale/shift the normal variables
        gauss_vars = 0
        if self.use_mu:
            gauss_vars = self.mu_np.reshape(1, 1, act_dim)
        gauss_vars = gauss_vars + normal_vars * self.std_np.reshape(1, 1, act_dim)
        # Step 3) We repeat `normal_vars` across the time dimension for
        #         keeping the exploratory noise low-frequencey
        noise_array = np.tile(gauss_vars, (1, time_steps, 1))
        if depth is None:
            noise_array = noise_array.reshape(rows, cols)

        return noise_array

    def update(self, da, A):
        """
        This function updates the internal exploration parameters

        Arguments:
          da (np.array): All of the collected noise sequences in a single array.
                         This should have a shape (batch, rows, cols).
                         (Remember: rows -> time steps,
                                    cols -> action dims)

          A (np.array): Corresponding advantages to each noise sequence.
                        This should have a shape (batch,).
        """
        mb_size = da.shape[0]
        rows = da.shape[1]
        cols = da.shape[2]

        a = da[:, 0, :].reshape(mb_size, cols)
        assert A.shape == (mb_size,)

        self.data_buffer.append((a, A))
        if len(self.data_buffer) > self.max_buff_len:
            self.data_buffer = self.data_buffer[-self.max_buff_len:]
        full_a_lst = [x[0] for _, x in enumerate(self.data_buffer)]
        full_A_lst = [x[1] for _, x in enumerate(self.data_buffer)]
        full_a = torch.cat(full_a_lst, dim=0)
        full_A = torch.cat(full_A_lst, dim=0)

        for ep_ in range(self.epochs):
            self.optimizer.zero_grad()
            dist = torch.distributions.normal.Normal(self.mu, torch.exp(self.log_std))
            log_pi = dist.log_prob(full_a).sum(dim=1)
            with torch.no_grad():
                old_log_pi = log_pi.detach()
            ratio = torch.exp(log_pi - old_log_pi)
            loss = - (ratio * full_A).mean()
            loss.backward()
            self.optimizer.step()

    def state_dict(self):
        return self.params

    def load_state_dict(self, sdict, strict=True):
        assert strict
        expected_keys = ['log_std']
        if self.use_mu:
            expected_keys.append('mu')
        assert sorted(tuple(sdict.keys())) == sorted(tuple(expected_keys))

        assert sdict['log_std'].shape == self.log_std.shape
        self.log_std.data = sdict['log_std']

        if self.use_mu:
            assert sdict['mu'].shape == self.mu.shape
            self.mu.data = sdict['mu']

class PPOAgent(object):
    def __init__(self, env_maker_fn, **kwargs):
        self.env_maker_fn = env_maker_fn
        self.env = self.env_maker_fn()
        self.reset(**kwargs)
        self.kwargs = copy.deepcopy(kwargs)

    def reset(self, **kwargs):
        self.actor_class = kwargs.get('actor_class', 'dnn')
        if self.actor_class == 'dnn':
            self.actor = Actor(self.env.observation_dim, self.env.action_dim, **kwargs)
        elif self.actor_class == 'rbf':
            self.actor = RBFActor(self.env.observation_dim, self.env.action_dim, **kwargs)
        self.critic = Critic(self.env.observation_dim, self.env.action_dim, **kwargs)

        net_init_f = kwargs.get('net_init_f', None)
        for name, p in self.actor.named_parameters():
            if net_init_f is None:
                if 'weight' in name:
                    # torch.nn.init.orthogonal_(p)
                    torch.nn.init.xavier_uniform_(p)
                elif 'bias' in name:
                    torch.nn.init.constant_(p, 0.)

                if actor_class == 'rbf':
                    if 'rbf_means' in name:
                        torch.nn.init.normal_(p,mean=0.0, std=1.0)
            else:
                net_init_f(name, p)

        for name, p in self.critic.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                torch.nn.init.constant_(p, 0.)

    def action_greedy(self, s):
        with torch.no_grad():
            (mu, std) = self.actor(torch.from_numpy(s).double())
            return mu.numpy()

    def action(self, s):
        with torch.no_grad():
            (mu, std) = self.actor(torch.from_numpy(s).double())
            dist = torch.distributions.normal.Normal(mu, std)
            a = dist.sample()
            return a.numpy()

    def greedy_actor_to_matlab(self, name='actor', ndigits=15, wtype='f'):
        """
        Exports greedy version of actor to matlab.

        If test=True, includes a function to report the maximum error between
        the greedy action computed in python (here) and in matlab.
        """
        actor_text = f'function y = {name}(x)\n' + \
                     f'% Returns greedy action from learned DNN policy.\n' + \
                     f'% size(x) = {self.env.observation_dim}\n' + \
                     f'% size(y) = {self.env.action_dim}\n' + \
                     f'y = fc3_mu(fc2(fc1(x)));\n' + \
                     f'end\n\n' + \
                     '% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n' + \
                     '% start of DNN layers for actor\n\n' + \
                     fc_to_matlab(self.actor.fc1, act='tanh', name='fc1', ndigits=ndigits, wtype=wtype) + '\n' + \
                     fc_to_matlab(self.actor.fc2, act='tanh', name='fc2', ndigits=ndigits, wtype=wtype) + '\n' + \
                     fc_to_matlab(self.actor.fc3_mu, act=None, name='fc3_mu', ndigits=ndigits, wtype=wtype) + '\n' + \
                     '% end of DNN layers for actor\n' + \
                     '% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'
        s = np.random.randn(self.env.observation_dim)
        a = self.action_greedy(s)
        s_matlab = numpy_to_matlab(np.reshape(s, (-1, 1)), ndigits=8, wtype='f')
        a_matlab = numpy_to_matlab(np.reshape(a, (-1, 1)), ndigits=8, wtype='f')
        test_text = f'function test_{name}()\n' + \
                    f'% Reports maximum absolute error between python and matlab,\n' + \
                    f'% for a particular input/output pair.\n' + \
                    f's = {s_matlab};\n' + \
                    f'a_from_python = {a_matlab};\n' + \
                    f'a_from_matlab = {name}(s);\n' + \
                    f'max_abs_err = max(abs(a_from_python - a_from_matlab));\n' + \
                    f'fprintf(1, "maximum absolute error between python and matlab (should be small): %g\\n", max_abs_err);\n' + \
                    f'end\n'
        return actor_text, test_text

    def value(self, s):
        with torch.no_grad():
            V = self.critic(torch.from_numpy(s).double())
            return V.item()

    def run_actor_greedy(self, policy_=None, steps_per_iteration=None, clip_action=False):
        assert self.env_type == 'stepper', 'roller not implemented yet.'
        policy_ = policy_ or self.action_greedy
        s = []
        a = []
        r = []
        s_next = self.env.reset()
        i_step = 0
        while True:
            if steps_per_iteration and i_step >= steps_per_iteration:
                break;
            s.append(s_next)
            a_next = policy_(s_next)
            a.append(a_next)
            s_next, r_next, done, _ = self.env.step(self.prepare_action(a_next, self.env, clip_action=clip_action))
            r.append(r_next)
            i_step += 1
            if done:
                break
        s = np.array(s)
        a = np.array(a)
        r = np.array(r)
        return s, a, r

    def L_clip(self, ratio, A, epsilon):
        return torch.min(A * ratio, A * torch.clamp(ratio, 1 - epsilon, 1 + epsilon))

    def broadcast_params(self):
        self.actor.broadcast()
        self.critic.broadcast()

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + '_actor')
        torch.save(self.critic.state_dict(), filename + '_critic')
        self.env.save(filename)

    def load(self, filename):
        self.actor = Actor(self.env.observation_dim, self.env.action_dim, **self.kwargs)
        self.critic = Critic(self.env.observation_dim, self.env.action_dim, **self.kwargs)
        self.actor.load_state_dict(torch.load(filename + '_actor'))
        self.critic.load_state_dict(torch.load(filename + '_critic'))

    def next_log_number(self, logdir_base):
        """Finds the number NNN for a logdir like <logdir_base>/NNN-XXX"""
        if not os.path.exists(logdir_base):
            return 1
        if not os.path.isdir(logdir_base):
            raise Exception(f'logdir {logdir_base} exists but is not a directory')
        highest = 0
        for filename in os.listdir(logdir_base):
            m = re.match('^([0-9]+)-', filename)
            if m:
                try:
                    n = int(m.group(1))
                    highest = max(highest, n)
                except:
                    pass
        return highest + 1

    def id2num(self, id):
        """Takes a job id like '23235.domain.net' and returns '23235', or None if the id doesn't start with a number."""
        if id is None:
            return None
        num = None
        m = re.match('^([0-9]+)', id)
        if m:
            num = m.group(1)
        return num

    def get_A_and_V_targ(self, r, V, V_bootstrap, gamma, lamb):
        # Get length of trajectory
        len = r.size
        # Create a copy of r and V, appended with V_bootstrap
        r = np.append(r, V_bootstrap)
        V = np.append(V, V_bootstrap)
        # Get advantage estimates
        # ... compute deltas ...
        A = r[:-1] + (gamma * V[1:]) - V[:-1]
        # ... compute advantages as reversed, discounted, cumulative sum of deltas ...
        for t in reversed(range(len - 1)):
            A[t] = A[t] + (gamma * lamb * A[t + 1])
        # Get value targets
        for t in reversed(range(len)):
            V[t] = r[t] + (gamma * V[t + 1])
        return A, V[:-1]

    def rollout(self, n_steps, gamma, lamb, just_one_rollout=False, shift_rewards=0):
        assert self.env_type == 'stepper', 'dont use this function for roller envs.'
        with torch.no_grad():
            s = np.zeros((n_steps, self.env.observation_dim))
            a = np.zeros((n_steps, self.env.action_dim))
            r = np.zeros(n_steps)
            log_pi = np.zeros(n_steps)
            V = np.zeros(n_steps)
            A = np.zeros(n_steps)
            V_targ = np.zeros(n_steps)

            t0 = 0
            s_next = self.env.reset()
            for t in range(n_steps):
                s[t, :] = s_next
                (mu_t, std_t) = self.actor(torch.from_numpy(s[t]).double())
                V[t] = self.critic(torch.from_numpy(s[t]).double())
                dist = torch.distributions.normal.Normal(mu_t, std_t)
                a[t] = dist.sample()
                log_pi[t] = dist.log_prob(torch.from_numpy(a[t]).double()).sum()
                (s_next, r[t], done, _) = self.env.step(a[t])
                if done:
                    r[t0:t + 1 - shift_rewards] = r[t0 + shift_rewards:t + 1]  # This is for compensating the actuator delay
                    A[t0:t + 1], V_targ[t0:t + 1] = self.get_A_and_V_targ(r[t0:t + 1], V[t0:t + 1], 0, gamma, lamb)
                    t0 = t + 1
                    s_next = self.env.reset()
                    # VERY IMPORTANT NOTE: We rely on the environment to have some
                    # maximum horizon. This is NOT standard with gym environments.
                    if just_one_rollout:
                        break
                elif t + 1 == n_steps:
                    V_next = self.critic(torch.from_numpy(s_next).double())
                    r[t0:t + 1 - shift_rewards] = r[
                                                  t0 + shift_rewards:t + 1]  # This is for compensating the actuator delay
                    A[t0:t + 1], V_targ[t0:t + 1] = self.get_A_and_V_targ(r[t0:t + 1], V[t0:t + 1], V_next, gamma, lamb)
                    s_next = self.env.reset()

            # in case of just_one_rollout we will have less data, so truncate
            s = s[:t + 1, :]
            a = a[:t + 1, :]
            r = r[:t + 1]
            log_pi = log_pi[:t + 1]
            V = V[:t + 1]
            A = A[:t + 1]
            V_targ = V_targ[:t + 1]

            return {
                's': torch.from_numpy(s),
                'a': torch.from_numpy(a),
                'r': torch.from_numpy(r),
                'log_pi': torch.from_numpy(log_pi),
                'V_targ': torch.from_numpy(V_targ),
                'A': torch.from_numpy(A),
                'V': torch.from_numpy(V),
            }

    def get_vine_test_places(self, gamma, sample_num, t, discount_visitation=True):
        t_ = max(t, 0)
        if discount_visitation:
            reset_indecis = np.random.geometric(p=1. - gamma, size=sample_num)
        else:
            reset_indecis = np.random.randint(0, high=t_+1, size=sample_num, dtype=int)
        reset_indecis = reset_indecis % (t_ + 1)
        return reset_indecis

    def prepare_action(self, action, env, clip_action=False):
        # Unfortunately, there is some sort of a sneakiness in the gym environments, where if you don't
        # clip your actions before passing them to the environemnt, you may get unnecessary negative rewards
        # and the this unnecessary negative reward has a devastating reward shaping effect.

        # This is quite seeable in the walker environment...
        if clip_action:
            return np.clip(action, env.action_space.low, env.action_space.high)
        else:
            return action

    def vine_rollout(self, n_steps, gamma, exploration_policy, env_rng_syncer, grp_rank_,
                     d1=1, d2=2, sample_num=10, efficient_reset=False,
                     discount_visitation=True, clip_action=False):
        assert self.env_type == 'stepper', 'dont use this function for roller envs.'
        assert efficient_reset == False
        assert d2 == 2
        with torch.no_grad():
            s = np.zeros((sample_num, n_steps, self.env.observation_dim))
            a = np.zeros((sample_num, n_steps, self.env.action_dim))
            r = np.zeros((sample_num, n_steps))
            Q = np.zeros((sample_num, n_steps))

            s_vine = np.zeros((sample_num, d1 * self.env.observation_dim)) # This mostly has states with exploratory actions
            s_vine_next = np.zeros((sample_num, d1 * self.env.observation_dim)) # This mostly has states with greedy actions
            a_vine = np.zeros((sample_num, d1 * self.env.action_dim)) # This mostly has exploratory actions
            a_vine_next = np.zeros((sample_num, d1 * self.env.action_dim)) # This mostly has greedy actions
            expl_noise = np.zeros((sample_num, d1 * self.env.action_dim)) # This has the additive noise to actions
            # Note: for d1 > 1 we do NOT have "expl_noise = a_vine - a_vine_next"!
            Q_vine = np.zeros(sample_num)
            eta_np = np.zeros(sample_num)

            randns_ = np.random.randn(sample_num, n_steps, self.env.action_dim)
            randns_ = torch.from_numpy(randns_).double()
            total_samps = 0
            reset_indecis = self.get_vine_test_places(gamma, sample_num, 10**10,
                                                      discount_visitation=discount_visitation)

            ### Error Note: d2>2 does not work properly at this point.
            ### I think (i.e., am not sure) the reason could be that reset_indecis are not synchronized in each group.
            ### This could explian why d1>1 works file, but d2>2 has problems; when d2=2, whatever d1 is, the group root won't
            ### make any purtarbations, and therefore not being in sync would not matter.
            ### Before fixing this issue, test the hypothesis and make sure that it is correct.

            #print(reset_indecis, flush=True)

            for sample_idx in range(sample_num):
                reset_idx = int(reset_indecis[sample_idx]) % (n_steps + 1 - d1 * (d2-1))
                env_rng_syncer(self.env)
                exploration_policy.set_d1(d1)
                grp_noise = exploration_policy.reset()
                assert grp_noise.shape == ((d2-1) * d1,  self.env.action_dim), \
                        f'{grp_noise.shape}, {d1}, {d2}, {self.env.action_dim}'
                rng_state_backup_ = copy.deepcopy(self.env.np_random.get_state())
                break_while = False
                while not break_while:
                    self.env.np_random.set_state(rng_state_backup_)
                    s_next = self.env.reset()
                    for t in range(n_steps):
                        s[sample_idx, t, :] = s_next
                        (mu_t, std_t) = exploration_policy(torch.from_numpy(s_next).double(), t, reset_idx)
                        next_a = mu_t + std_t * randns_[sample_idx, t, :]
                        a[sample_idx, t, :] = next_a
                        (s_next, r[sample_idx, t], done, _) = self.env.step(self.prepare_action(next_a.numpy(), self.env, clip_action=clip_action))
                        done = done or (t == (n_steps-1))
                        if done:
                            #print(f'rank {rank}: t= {t}', flush=True)
                            # Warning: t may not be the same among the group workers at this point
                            if t < reset_idx:
                                # The reset idx was too late, and the env does not seem stable long enough.
                                # We just have to exactly repeat the trajectory with a more plausible reset idx
                                reset_idx = int(reset_indecis[sample_idx]) % (t + 2 - d1 * (d2-1))
                                reset_indecis[sample_idx] = reset_idx
                                break_while = False
                                break
                            else:
                                total_samps += t
                                break_while = True
                                r[sample_idx, t + 1:] = 0
                                s[sample_idx, t + 1:, :] = s_next
                                # assigning a after termination is such an edge case that I am going to ignore.
                                for useless_t in range(t + 1, reset_idx + d1 * (d2-1)):
                                    (mu_t, std_t) = exploration_policy(torch.from_numpy(s_next).double(), useless_t, reset_idx)
                                    next_a = mu_t + std_t * randns_[sample_idx, useless_t, :]
                                    a[sample_idx, useless_t, :] = next_a
                                reset_indecis[sample_idx] = reset_idx
                                break

                Q[sample_idx, :] = mycumprodsum_rev(r[sample_idx, :], gamma)

                shft_ = max(0, grp_rank_-1) * d1
                st_idx = reset_idx + shft_
                end_idx = st_idx + d1
                st_idx_next = reset_idx + min(d2-2, grp_rank_) * d1
                end_idx_next = st_idx_next + d1
                s_vine[sample_idx, :] = s[sample_idx, st_idx: end_idx, :].reshape(-1)
                a_vine[sample_idx, :] = a[sample_idx, st_idx: end_idx, :].reshape(-1)
                s_vine_next[sample_idx, :] = s[sample_idx, st_idx_next: end_idx_next, :].reshape(-1)
                a_vine_next[sample_idx, :] = a[sample_idx, st_idx_next: end_idx_next, :].reshape(-1)
                Q_vine[sample_idx] = Q[sample_idx, reset_idx]
                eta_np[sample_idx] = Q[sample_idx, 0]
                expl_noise[sample_idx, :] = grp_noise[shft_:(shft_+d1), :].reshape(-1)

            return {
                's_vine': torch.from_numpy(s_vine),
                'a_vine': torch.from_numpy(a_vine),
                's_vine_next': torch.from_numpy(s_vine_next),
                'a_vine_next': torch.from_numpy(a_vine_next),
                'reset_times': torch.from_numpy(reset_indecis),
                'eta': torch.tensor(eta_np.mean()),
                'Q_vine': torch.from_numpy(Q_vine),
                #'s_all': torch.from_numpy(s.reshape(sample_num * n_steps, self.env.observation_dim)),
                #'Q_all': torch.from_numpy(Q),
                'undiscounted_payoff': torch.tensor(r.sum()/sample_num),
                'samples_count': total_samps,
                'expl_noise': torch.from_numpy(expl_noise),
            }

    def compile_vine_datasets(self, data_vine, grp_rank_, d1, d2, sample_num, rank, size):
        assert data_vine['s_vine'].shape[0] == sample_num
        #expl_noise = np.zeros((sample_num, d1 * self.env.action_dim))
        with torch.no_grad():
            comm = MPI.COMM_WORLD
            num_grps = size // d2
            data_vine_root = dict()
            for gather_key in ['s_vine', 'a_vine', 'a_vine_next', 'expl_noise']: #'s_vine_next',
                data_vine_root[gather_key] = gather_rows(data_vine[gather_key])
            for gather_key in ['Q_vine', 'reset_times']:
                data_vine_root[gather_key] = gather_rows(data_vine[gather_key].reshape(-1, 1))

            data_vine_root['eta'] = reduce_average(data_vine['eta'] * float(rank%d2==0)) * d2
            data_vine_root['undiscounted_payoff'] = reduce_average(data_vine['undiscounted_payoff'] * float(rank%d2==0)) * d2
            data_vine_root['samples_count'] = comm.reduce(data_vine['samples_count'], op=MPI.SUM, root=0) or 0

            out_data = dict()
            if rank == 0:
                for gather_key in  ['s_vine', 'a_vine', 'a_vine_next', 'expl_noise', 'Q_vine']:#'s_vine_next',
                    data_vine_root[gather_key] = data_vine_root[gather_key].reshape(size, sample_num, -1)#d1 * self.env.observation/action/1_dim

                s_vine = np.zeros((sample_num * (d2-1) * num_grps, d1 * self.env.observation_dim))
                a_vine_test = np.zeros((sample_num * (d2-1) * num_grps, d1 * self.env.action_dim))
                a_vine_orig = np.zeros((sample_num * (d2-1) * num_grps, d1 * self.env.action_dim))
                expl_noise = np.zeros((sample_num * (d2-1) * num_grps, d1 * self.env.action_dim))
                A_vine = np.zeros((sample_num * (d2-1) * num_grps))

                non_grp_root_ranks = [x_ for x_ in range(size) if (x_%d2)>0]

                row_id = 0
                for sample_idx in range(sample_num):
                    for rnk in non_grp_root_ranks:
                        s_vine[row_id, :] = data_vine_root['s_vine'].numpy()[rnk, sample_idx, :]
                        a_vine_test[row_id, :] = data_vine_root['a_vine'].numpy()[rnk, sample_idx, :]
                        expl_noise[row_id, :] = data_vine_root['expl_noise'].numpy()[rnk, sample_idx, :]
                        a_vine_orig[row_id, :] = data_vine_root['a_vine_next'].numpy()[rnk-1, sample_idx, :]
                        A_vine[row_id] = data_vine_root['Q_vine'].numpy()[rnk, sample_idx, :] - data_vine_root['Q_vine'].numpy()[rnk-1, sample_idx, :]
                        row_id += 1

                with torch.no_grad():
                    out_data['eta'] = data_vine_root['eta']
                    out_data['undiscounted_payoff'] = data_vine_root['undiscounted_payoff']
                    out_data['samples_count'] = data_vine_root['samples_count']
                    #out_data['reset_times'] = data_vine_root['reset_times'].reshape(-1)
                    out_data['s_vine'] = torch.from_numpy(s_vine).double()
                    out_data['a_vine_orig'] = torch.from_numpy(a_vine_orig).double()
                    out_data['a_vine_test'] = torch.from_numpy(a_vine_test).double()
                    out_data['expl_noise'] = torch.from_numpy(expl_noise).double()
                    out_data['A_vine'] = torch.from_numpy(A_vine).double()

            return out_data

    def decouple_compiled_datasets(self, data_vine_root, grp_rank_, d1, d2, sample_num, rank, size):

        out_data = dict()
        comm = MPI.COMM_WORLD

        # data_vine_root is full of stuff for root, and otherwise is an empty dictionary.
        root_bcast_key = lambda key, dtype_: torch.tensor(comm.bcast(data_vine_root.get(key, torch.tensor(0)).numpy().astype(dtype_), root=0))
        out_data['eta'] = root_bcast_key('eta', np.float64)
        out_data['undiscounted_payoff'] = root_bcast_key('undiscounted_payoff', np.float64)
        out_data['samples_count'] = comm.bcast(int(data_vine_root.get('samples_count', 0)), root=0)


        def scatter_data(key_, double_dtype=False, grouping=d2-1, repeats=d2, ravel_finally=False):
            with torch.no_grad():
                if rank == 0:
                    np_var = data_vine_root[key_].numpy()
                    if np_var.shape[0] % size == 0:
                        # The data is easily dividable among ranks. No need for further complication/extra computational overhead.
                        np_var_tiled = np_var.reshape(size, -1, *np_var.shape[1:])
                    else:
                        # We'll have to repeat the data, to keep the convention that all ranks should do something!
                        expected_mpi_size = ((repeats * np_var.shape[0]) // (grouping * sample_num))
                        assert size == expected_mpi_size
                        np_var_tiled = repeat_and_divide(np_var=np_var, grouping=grouping, repeats=repeats, mpi_size=expected_mpi_size)
                else:
                    np_var_tiled = None
                scattered_np = comm.scatter(np_var_tiled, root=0)

                if ravel_finally:
                    # In case we'd like to
                    scattered_np = scattered_np.reshape(-1)

                scattered_tensor = torch.from_numpy(scattered_np)
                if double_dtype:
                    scattered_tensor = scattered_tensor.double()
            return scattered_tensor

        #out_data['reset_times'] = scatter_data('reset_times', double_dtype=False, grouping=1, repeats=1, ravel_finally=True)
        out_data['s_vine'] = scatter_data('s_vine', double_dtype=True, grouping=d2-1, repeats=d2, ravel_finally=False)
        out_data['a_vine_orig'] = scatter_data('a_vine_orig', double_dtype=True, grouping=d2-1, repeats=d2, ravel_finally=False)
        out_data['a_vine_test'] = scatter_data('a_vine_test', double_dtype=True, grouping=d2-1, repeats=d2, ravel_finally=False)
        out_data['A_vine'] = scatter_data('A_vine', double_dtype=True, grouping=d2-1, repeats=d2, ravel_finally=True)

        return out_data

    def env_rollout_vine(self, n_steps, gamma, policy, explorer, d1=1, sample_num=10,
                         discount_visitation=True):

        assert sample_num % 2 == 0, 'Every vine takes two trajectories'

        vine_samp_num = sample_num // 2
        # 1) Sampling the vine time-step indecis
        reset_indecis = self.get_vine_test_places(gamma, vine_samp_num, n_steps - d1,
                                                  discount_visitation)

        # 2) Sampling the exploration noise
        expl_noise = explorer(d1, self.env.action_dim, np.random, depth=vine_samp_num)

        # 3) Sending the policy to the environement for automatic rollouts
        self.env.set_torch_policy(policy)

        # 4) Asking for vine rollouts
        #  --> the env will sample thie init states and take care of synchronizations.
        env_out = self.env.vine_lite(traj_num=vine_samp_num, n_steps=n_steps, gamma=gamma,
                                     expl_steps=d1, reset_times=reset_indecis,
                                     expl_noise=expl_noise)

        env_out['expl_noise'] = expl_noise

        ################ Gathering the MPI Samples #################
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        keys2d = ["obs_greedy", "action_greedy", "action_vine", "expl_noise"]
        keys1d = ["Q_greedy", "eta_greedy", "return_greedy",
                  "Q_vine", "eta_vine", "return_vine", "done_steps"]

        gathered_data = dict()
        for _, (key, val) in enumerate(sorted(env_out.items())):
            if key in keys2d:
                gathered_data[key] = gather_np(val.reshape(vine_samp_num,-1))
            elif key in keys1d:
                gathered_data[key] = gather_np(val.reshape(vine_samp_num,1))
            else:
                raise Exception(f'I dont know how to gather {key}')


        ############# Processing the Output Dictionary #############


        out_data = dict()
        if rank == 0:

            out_data['s_vine'] = gathered_data['obs_greedy']
            out_data['a_vine_orig'] = gathered_data['action_greedy']
            out_data['a_vine_test'] = gathered_data['action_vine']
            out_data['expl_noise'] = gathered_data['expl_noise']
            out_data['A_vine'] = (gathered_data['Q_vine'].reshape(-1) -
                                  gathered_data['Q_greedy'].reshape(-1))
            out_data = {key: torch.from_numpy(val) for key, val in out_data.items()}

            out_data['eta'] = torch.tensor(gathered_data['eta_greedy'].mean())
            out_data['undiscounted_payoff'] = torch.tensor(gathered_data['return_greedy'].mean())
            out_data['samples_count'] = gathered_data['done_steps'].sum().item()

        return out_data

    def seed(self, my_seed):
        random.seed(my_seed)
        np.random.seed(my_seed)
        torch.manual_seed(my_seed)
        self.env.seed(my_seed)

    def train(self, nondefault_options={}):
        ######################################################################
        # options handling

        options = {
            'logdir_base': None,
            'log_prefix': 'ppo',
            'gamma': 0.99,
            'lamb': 0.95,
            'gamma_rate': None,
            'lamb_rate': None,
            'number_of_steps': 1_000_000_000,
            'steps_per_iteration_per_process': 4000,
            'steps_per_epoch': None,
            'steps_per_snapshot': 20_000,
            'number_of_epochs': 80,
            'epsilon': 0.2,
            'max_KL_per_iteration': 0.01,
            'std0': None,
            'std1': None,
            'actor_lr': 3e-4,
            'critic_lr': 1e-3,
            'seed': int.from_bytes(os.urandom(4), byteorder='little'),
            'output_histograms': True,
            'pbs_jobname': os.environ.get('PBS_JOBNAME', None),
            'pbs_o_workdir': os.environ.get('PBS_O_WORKDIR', None),
            'pbs_queue': os.environ.get('PBS_QUEUE', None),
            'pbs_o_host': os.environ.get('PBS_O_HOST', None),
            'pbs_jobid': os.environ.get('PBS_JOBID', None),
            'pbs_jobnum': self.id2num(os.environ.get('PBS_JOBID', None)),
            'shift_rewards': 0,
            'alg': 'ppo',
            'reseed_each_iter': False,
            'vine_params': None,
            # 'vine_params' : dict(samples_per_traj = 10, action_disturbance = 0.01,
            #                      method='finite_diff', 'max_payoff_increase':0.1,
            #                      'efficient_reset':True)
            'transfer_data_to_root': False,
            'wtrpo_opt_params': {'type': 'lbfgs'},
            'identity_stabilization': 1e-2,
            'wppo_params': dict(C1=1e6, C2=1e6),
            'environment_type': 'stepper',
            'ploting_environment': None,
            'greedy_plot': True, 'noisy_plot': True,
            'export_baseline_format': False,
            'logdit_tail': None,
            'adaptive_c_params': None,  # options['adaptive_c_params'] = dict(min_c_w2=1e-2, smoothing_factor=0.9)
            'verbose': False,
            'hyperparam_resizer': None,
            'enable_tb': True,
            'out_file': None,
            'optim_maker': None,
            'obs_plot_lists': None,
            'act_plot_lists': None,
            'walltime_hrs': 72,
        }
        extra_keys = nondefault_options.keys() - options.keys()
        if len(extra_keys) > 0:
            raise Exception(f'Unknown options: {extra_keys}')
        options.update(nondefault_options)

        self.env_type = options['environment_type']
        assert self.env_type in ('stepper', 'roller')

        ######################################################################
        # MPI init

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        if rank == 0:
            print(f'{size} MPI processes')

        ######################################################################
        # RNG seeding

        root_seed = comm.bcast(options['seed'], root=0)
        my_seed = root_seed + rank
        self.seed(my_seed)
        # If sampling is vine, we will also reseed properly later.

        ######################################################################
        # discount rate parameter calculations

        # if we have rate parameters then compute the discrete-time analogues
        if options['gamma_rate'] is not None:
            options['gamma'] = np.exp(-options['gamma_rate'] * self.env.get_timestep())
        if options['lamb_rate'] is not None:
            options['lamb'] = np.exp(-options['lamb_rate'] * self.env.get_timestep())

        ######################################################################
        # WXPO Settings
        alg_ = options['alg']
        alg_is_greedy_ = alg_ in ['wtrpo', 'wppo']

        backup_actor = copy.deepcopy(self.actor)
        identity_stabilization = options['identity_stabilization']
        payoff_grad_coeff = 1.

        if alg_ == 'wtrpo':
            C_w2, C_wg = options['wtrpo_opt_params'].get('Cw2_Cwg', (None, None))
            assert options['wtrpo_opt_params']['type'] == 'cg'
            assert (C_w2 is not None)
            payoff_grad_ = tuple(torch.zeros_like(w_) for w_ in self.actor.parameters())
            neg_gain_grad_ = tuple(torch.zeros_like(w_) for w_ in self.actor.parameters())
            payoff_inv_dir = tuple(torch.zeros_like(w_) for w_ in self.actor.parameters())

            self.closure = AClosure(policy_params=tuple(self.actor.parameters()),
                                    identity_stabilization=identity_stabilization,
                                    mpi_enabled=False,
                                    GSq2WSq_ratio=C_wg/C_w2,
                                    average_over_rows=True,
                                    obs2act_jac='exact')
            # average_over_rows = True is one of the most important changes made in version 3

            options['wtrpo_opt_params'].setdefault('stop_criterion', ('eps', 1e-8))
            line_search_coeffs = options['wtrpo_opt_params'].get('line_search_coeffs', None)
            if line_search_coeffs is not None:
                line_search_coeffs = np.array(line_search_coeffs).reshape(-1)
                line_search_coeffs.sort()
                lsc_full, ls_seeds_full = prep_lsc(line_search_coeffs, size)
                ls_rounds = len(lsc_full)//size
                a, b = (rank*ls_rounds), ((rank+1)*ls_rounds)
                my_ls_coeffs, my_ls_seeds_base = lsc_full[a:b], ls_seeds_full[a:b]

            max_w_ = float(options['wtrpo_opt_params'].get('max_w', 10))
        elif alg_ == 'wppo':
            C_wg, C_w2 = options['wppo_params']['C1'], options['wppo_params']['C2']
            max_w_ = options['wppo_params'].get('max_w', 0.01)

        identity_enabled_ = not((options['identity_stabilization'] is None) or
                                (options['identity_stabilization'] == 0))
        if alg_ in ['wppo' , 'wtrpo']:
            gain_term_enabled_ = not ((C_wg is None) or (C_wg == 0))
            w2_term_enabled_ = not ((C_w2 is None) or (C_w2 == 0))
        else:
            gain_term_enabled_ = True
            w2_term_enabled_ = True

        ######################################################################
        # Vine Setting
        policy = self.actor

        clip_action = False
        assert options['vine_params'].get('method', 'finite_diff') in ['finite_diff']

        vine_sample_num = options['vine_params'].get('samples_per_traj', 10)
        #d1, d2, exp_noise_generator

        d1_vine_list = options['vine_params'].get('d1', 1)
        if isinstance(d1_vine_list, int):
            d1_vine_list = [d1_vine_list]
        d1_vine_cycler = cycle(d1_vine_list)
        d2_vine = options['vine_params'].get('d2', 2)
        #d_vine = d1_vine * d2_vine
        my_grp_rank = int(rank % d2_vine)
        my_grp_root = d2_vine * int(rank / d2_vine)
        my_grp_id = int(rank / d2_vine)
        clip_action = options['vine_params'].get('clip_action', False)
        explorer = options['vine_params'].get('exp_noise_generator', None)
        explorer_needs_update = hasattr(explorer, 'update')
        post_proc = options['vine_params'].get('exp_act_post_proc', None)

        assert size% d2_vine == 0
        vine_exploration_policy = Vine_Policy(my_grp_rank, self.env.action_dim,
                                              options, self.actor,
                                              d2=d2_vine, np_random=np.random,
                                              exp_noise_generator=explorer,
                                              exp_act_post_proc=post_proc)

        def env_rng_syncer(env_):
            rng_state_init_ = copy.deepcopy(env_.np_random.get_state())
            mtstr, keys, pos, has_gauss, cached_gaussian = rng_state_init_
            dtypes = (keys.dtype, type(pos), type(has_gauss), type(cached_gaussian))
            rng_state_np = np.concatenate([keys.reshape(-1), [pos, has_gauss, cached_gaussian]])
            n_cols = rng_state_np.size

            rng_state_np_root = gather_np(rng_state_np.reshape(1,-1))
            if rank==0:
                for i_ in range(size):
                    last_worker_idx = d2_vine * int(i_ // d2_vine) + d2_vine - 1
                    rng_state_np_root[i_, :] = rng_state_np_root[last_worker_idx, :]

            post_rng_state_np = np.empty_like(rng_state_np)
            comm.Scatter(rng_state_np_root, post_rng_state_np, root=0)

            keys_ = post_rng_state_np[:-3].astype(dtypes[-4])
            pos_ = dtypes[1](post_rng_state_np[-3])
            has_gauss_ = dtypes[2](post_rng_state_np[-2])
            cached_gaussian_ = dtypes[3](post_rng_state_np[-1])
            post_rng_state = (mtstr, keys_, pos_, has_gauss_, cached_gaussian_)
            env_.np_random.set_state(post_rng_state)
            return env_

        def policy(state_tensor):
            mu_, std_ = self.actor(state_tensor)
            return mu_, std_ * 0.

        my_seed = root_seed + my_grp_id
        self.seed(my_seed)
        is_params = options['vine_params'].get('is_params', None)
        if is_params is not None:
            self.time_hist = np.zeros(options['steps_per_iteration_per_process'])
            self.act_dim_hist = np.zeros(self.env.action_dim)

        self.closure.set_policy(policy)
        ######################################################################
        # set up tensorboard logging

        if options['enable_tb'] and rank == 0:

            if options['logdir_base'] is None:
                if 'OPENAI_LOGDIR' in os.environ:
                    options['logdir_base'] = os.environ['OPENAI_LOGDIR']
                if 'WXPO_LOGDIR_BASE' in os.environ:
                    options['logdir_base'] = os.environ['WXPO_LOGDIR_BASE']
            if options['logdir_base'] is None:
                options['logdir_base'] = 'logdir'

            if options['logdit_tail'] == '':
                logdir_tail = ''
            else:
                log_time = datetime.datetime.now().isoformat(timespec='seconds')
                log_number = self.next_log_number(options['logdir_base'])
                logdir_tail = str(log_number)
                logdir_tail += '-' + options['log_prefix']
                if options['pbs_jobname'] is not None:
                    logdir_tail += '-' + options['pbs_jobname']
                if options['pbs_jobnum'] is not None:
                    logdir_tail += '-' + options['pbs_jobnum']
                logdir_tail += '-' + log_time

            logdir = os.path.join(options['logdir_base'], logdir_tail)
            if options['export_baseline_format']:
                monitor_dict = dict()
                progress_dict = dict()
                monitor_file = os.path.join(logdir, '0.0.monitor.csv')
                monitor_file_last_mod_time = time.time()
                progress_file = os.path.join(logdir, 'progress.csv')
            try:
                writer = SummaryWriter(logdir, flush_secs=10)
                progress_train_dict = dict()
                progress_train_file = os.path.join(logdir, 'progress_train.csv')
            except OSError as e_:
                # For Windows, this logdir format cannot be a name folder, and so we will just make it super smaller
                logdir = os.path.join(options['logdir_base'], options['log_prefix'].replace('-', '_').replace(':', '_'))
                writer = SummaryWriter(logdir, flush_secs=10)



        ######################################################################
        # log all the options and metadata as "text" in tensorboard

        if options['enable_tb'] and rank == 0:
            train_options_string = ''.join([f'* {key} = {value}\n' for key, value in options.items()])
            writer.add_text('train_options', train_options_string, 0)
            train_nondefault_options_string = ''.join(
                [f'* {key} = {value}\n' for key, value in nondefault_options.items()])
            writer.add_text('train_nondefault_options', train_nondefault_options_string, 0)
            env_options_string = ''.join([f'* {key} = {value}\n' for key, value in self.env.get_options().items()])
            writer.add_text('env_options', env_options_string, 0)
            env_nondefault_options_string = ''.join(
                [f'* {key} = {value}\n' for key, value in self.env.get_nondefault_options().items()])
            writer.add_text('env_nondefault_options', env_nondefault_options_string, 0)
            for key, value in self.env.get_metadata().items():
                writer.add_text(key, f'<pre>\n{value}\n<pre>\n', 0)
            writer.add_text('number of MPI processes', f'{size}', 0)  # FIXME: could handle this better

        ######################################################################
        # arrays of output data from training

        rewards = []
        rewards_dict = dict()
        losses_dict = dict()
        losses_clip = []
        losses_V = []
        stds = []
        times_sample = []
        times_opt = []
        epochs = []

        rewards_dict['undiscounted'] = []

        ######################################################################
        # main training loop
        if options['optim_maker'] is None:
            options['optim_maker'] = lambda params_, lr_: torch.optim.Adam(params_, lr=lr_)
        actor_optimizer = options['optim_maker'](self.actor.parameters(), options['actor_lr'])
        critic_optimizer = options['optim_maker'](self.critic.parameters(), options['critic_lr'])

        self.broadcast_params()

        n_steps = 0
        steps_since_last_snapshot = 0
        number_of_iterations = options['number_of_steps'] // options['steps_per_iteration_per_process'] // size
        number_of_iterations = number_of_iterations // (1 + vine_sample_num)

        loop_start_time = time.time()
        max_seconds = 3600. * options['walltime_hrs']

        iter = -1
        while n_steps <= options['number_of_steps']:
            iter = iter + 1

            latest_time = time.time()
            walltime_break = comm.bcast((latest_time-loop_start_time) > max_seconds, root=0)
            if walltime_break:
                break

            if rank == 0:
                print(f'iteration {iter} of {number_of_iterations}, n_steps = {n_steps}')

            # Reseeding for reducing variance ###############################
            if options['reseed_each_iter']:
                self.env.seed(my_seed)

            # Anneal std if necessary #######################################

            if options['std0'] is not None and options['std1'] is not None:
                alpha = iter / (number_of_iterations - 1)
                std = (1 - alpha) * options['std0'] + alpha * options['std1']
                self.actor.fixed_std = torch.tensor([std], dtype=torch.double)

            # Sampling ######################################################

            start_time = time.time()
            d1_vine = next(d1_vine_cycler)
            discount_visitation_ = options['vine_params'].get('discount_visitation', True)
            if self.env_type == 'stepper':
                efficient_reset_ = options['vine_params'].get('efficient_reset', True)
                data_vine_rollout = self.vine_rollout(options['steps_per_iteration_per_process'],
                                                      options['gamma'], vine_exploration_policy,
                                                      env_rng_syncer, my_grp_rank,
                                                      d1_vine, d2_vine, vine_sample_num,
                                                      efficient_reset=efficient_reset_,
                                                      discount_visitation=discount_visitation_,
                                                      clip_action=clip_action)

                data_vine = self.compile_vine_datasets(data_vine_rollout, my_grp_rank, d1_vine,
                                                       d2_vine, vine_sample_num, rank, size)
            elif self.env_type == 'roller':
                data_vine = self.env_rollout_vine(options['steps_per_iteration_per_process'],
                                                  options['gamma'], self.actor, explorer,
                                                  d1=d1_vine, sample_num=vine_sample_num,
                                                  discount_visitation=discount_visitation_)
            else:
                raise Exception(f'Unknown self.env_type {self.env_type}')

            with torch.no_grad():
                if rank == 0:
                    correction_ratio = (1. - options['gamma'])
                    correction_ratio /= (1. - (options['gamma'] ** options['steps_per_iteration_per_process']))
                    rewards.append(correction_ratio * data_vine['eta'].item())
                    rewards_dict['undiscounted'].append(data_vine['undiscounted_payoff'].item())

            if rank == 0:
                delta_steps = data_vine['samples_count'] or 0
            else:
                delta_steps = 0

            delta_steps = comm.bcast(delta_steps, root=0)
            n_steps += delta_steps
            steps_since_last_snapshot += delta_steps
            end_time = time.time()
            times_sample.append(end_time - start_time)

            # Pre optimization ##################################################
            start_time = time.time()
            iter_losses_dict = dict()
            iter_losses_clip = []
            iter_losses_V = []
            iter_stds = []
            epoch = 0

            # Actor optimization ##################################################
            if alg_ == 'ppo':
                for epoch in range(options['number_of_epochs']):
                    if options['steps_per_epoch'] is None or options['steps_per_epoch'] >= data['s'].shape[0]:
                        s = data['s']
                        a = data['a']
                        old_log_pi = data['log_pi']
                        V_targ = data['V_targ']
                        A = data['A']
                    else:
                        rand_ind = np.random.choice(data['s'].shape[0], options['steps_per_epoch'])
                        s = data['s'][rand_ind, :]
                        a = data['a'][rand_ind, :]
                        old_log_pi = data['log_pi'][rand_ind]
                        V_targ = data['V_targ'][rand_ind]
                        A = data['A'][rand_ind]

                    actor_optimizer.zero_grad()
                    (mu, std) = self.actor(s)
                    dist = torch.distributions.normal.Normal(mu, std)
                    log_pi = dist.log_prob(a).sum(dim=1)
                    ratio = torch.exp(log_pi - old_log_pi)
                    loss_clip = -self.L_clip(ratio, A, options['epsilon']).mean()
                    iter_losses_clip.append(reduce_average(loss_clip).item())
                    iter_stds.append(reduce_average(std.mean()).item())
                    loss_clip.backward()

                    self.actor.grad_reduce_average()
                    if rank == 0:
                        actor_optimizer.step()
                    self.actor.broadcast()

                    with torch.no_grad():
                        KL = torch.mean(old_log_pi - log_pi)
                        global_KL = reduce_average(KL)
                        do_break = comm.bcast(global_KL > 1.5 * options['max_KL_per_iteration'], root=0)
                        if do_break:
                            break

            elif alg_ == 'wppo':

                # Getting a network backup
                backup_actor.load_state_dict(self.actor.state_dict())

                for epoch in range(options['number_of_epochs']):
                    if rank > 0:
                        break

                    ######################################################################
                    # Computing Payoff Gradient
                    self.actor.zero_grad()
                    s = data_vine['s_vine']
                    a_test = data_vine['a_vine_test']
                    a_orig = data_vine['a_vine_orig']
                    A = data_vine['A_vine']

                    (mu, std) = policy(s)
                    a_disturbance = mu - a_orig
                    a_test_disturbance = a_test - a_orig
                    A_coeff = torch.sum(a_disturbance * a_test_disturbance, dim=1) / torch.sum(
                        a_test_disturbance * a_test_disturbance, dim=1)
                    policy_adv_vec = A_coeff * A
                    if options['wppo_params'].get('same_sign_adv', False):
                        policy_adv_vec = torch.where(policy_adv_vec >= 0, policy_adv_vec, policy_adv_vec * 0.001)
                    policy_adv = torch.mean(policy_adv_vec)

                    policy_adv = payoff_grad_coeff * policy_adv
                    #C1, C2 = options['wppo_params']['C1'], options['wppo_params']['C2']
                    C1, C2 = C_wg, C_w2
                    bs = s.shape[0]

                    if gain_term_enabled_ or w2_term_enabled_:
                        if epoch == 0:
                            mu_fixed = mu.clone().detach()
                            std_fixed = std.clone().detach()

                        mu_diff_sq = ((mu - mu_fixed) ** 2) / 2
                        std_diff_sq = ((std - std_fixed) ** 2) / 2

                        # Some Shaping Sanity Checks
                        if len(std_diff_sq.shape) == 2:
                            assert (std_diff_sq.shape == mu_diff_sq.shape) or (
                                    std_diff_sq.shape[0] == 1 and std_diff_sq.shape[1] == mu_diff_sq.shape[1])
                        else:
                            std_diff_sq = std_diff_sq.reshape(1, -1)
                            assert std_diff_sq.shape[1] == mu_diff_sq.shape[1]

                        wass_sq_vec = mu_diff_sq + std_diff_sq

                    if gain_term_enabled_:
                        eps_finite_diff = 1e-6
                        #obs_dim = self.env.observation_dim
                        obs_dim = s.shape[1]
                        obs_eye_mat = torch.eye(obs_dim, dtype=torch.double)
                        full_gain_sq_vec = 0.
                        for i in range(obs_dim):
                            s_new = s + obs_eye_mat[i] * eps_finite_diff
                            mu_new, std_new = policy(s_new)
                            full_gain_sq_vec += ((mu_new - mu) ** 2)
                        full_gain_sq_vec = full_gain_sq_vec / (eps_finite_diff ** 2)
                        full_gain_vec = torch.sqrt(full_gain_sq_vec)

                        if epoch == 0:
                            full_gain_vec_orig = full_gain_vec.clone().detach()

                        sqrt_stab = 1e-1
                        wass_vec = torch.sqrt(wass_sq_vec + sqrt_stab) - sqrt_stab**0.5

                        gain_term = C1 * torch.sum(full_gain_vec * wass_vec) / bs
                    else:
                        gain_term = torch.tensor(0.).double()

                    if identity_enabled_:
                        id_stab_loss = 0.
                        for p_, old_p_ in zip(self.actor.parameters(), backup_actor.parameters()):
                            e_ = p_ - old_p_.detach()
                            id_stab_loss += torch.sum(e_ ** 2)
                        id_term = id_stab_loss * identity_stabilization / 2
                    else:
                        id_term = torch.tensor(0.).double()

                    if w2_term_enabled_:
                        w2_val = torch.sum(wass_sq_vec) / bs
                    else:
                        w2_val = torch.tensor(0.).double()
                    w2_term = C2 * w2_val

                    max_objective = policy_adv - gain_term - w2_term - id_term
                    min_objective = -1. * max_objective
                    min_objective.backward()
                    if rank == 0:
                        actor_optimizer.step()

                    for name, val in [('w2_term', w2_term), ('w2_val', w2_val),
                                      ('gain_term', gain_term), ('id_term', id_term)]:
                        iter_losses_dict.setdefault(name, [])
                        iter_losses_dict[name].append(val.item())
                    iter_losses_clip.append(max_objective.item())
                    iter_stds.append(std.mean().item())


                    with torch.no_grad():
                        do_break = (w2_val) > (max_w_**2)
                        if do_break:
                            break


                if options['verbose'] and rank == 0:
                    policy_adv_, gain_term_, w2_term_, id_term_, max_objective_ = policy_adv.item(), gain_term.item(), w2_term.item(), id_term.item(), max_objective.item()
                    print(f'\t policy_adv: {policy_adv_}')
                    print(f'\t gain_term: {gain_term_} \t %.2f' % (100 * np.abs(gain_term_) / np.abs(policy_adv_)))
                    print(f'\t w2_term: {w2_term_} \t\t %.2f' % (100 * np.abs(w2_term_) / np.abs(policy_adv_)))
                    print(f'\t id_reg: {id_term_} \t\t %.2f' % (100 * np.abs(id_term_) / np.abs(policy_adv_)))
                    print(f'\t max_objective: {max_objective_}', flush=True)

                self.actor.broadcast()

            elif alg_ == 'wtrpo':
                # Getting a network backup
                backup_actor.load_state_dict(self.actor.state_dict())
                ######################################################################
                if rank == 0:
                    self.actor.zero_grad()

                    s = data_vine['s_vine']
                    a_test = data_vine['a_vine_test']
                    a_orig = data_vine['a_vine_orig']
                    A = data_vine['A_vine']

                    (mu, std) = policy(s)
                    a_disturbance = mu - a_orig
                    a_test_disturbance = a_test - a_orig
                    constant_linear_advantage_coeffs = A / torch.sum(a_test_disturbance * a_test_disturbance, dim=1)
                    candidate_policy_term =  torch.sum(a_disturbance * a_test_disturbance, dim=1)
                    policy_adv = torch.mean(candidate_policy_term * constant_linear_advantage_coeffs)
                    policy_adv = policy_adv * payoff_grad_coeff
                    policy_adv.backward()

                    # Collecting The gradient and storing it
                    for w_, g_ in zip(self.actor.parameters(), payoff_grad_):
                        if w_.grad is None:
                            g_.zero_()
                        else:
                            g_.copy_(w_.grad.data)

                    mu_fixed, std_fixed = (mu.clone().detach(), std.clone().detach())

                    check_nan(net_=self.actor)

                    self.closure.set_action_dist_params(s)

                    if options['wtrpo_opt_params'].get('adaptive_c', False):
                        with torch.no_grad():
                            w2_weights = torch.abs(constant_linear_advantage_coeffs)
                            if options['wtrpo_opt_params'].get('min_adaptive_c', None) is not None:
                                min_adaptive_c = options['wtrpo_opt_params']['min_adaptive_c']
                                small_c_ratio = (w2_weights <= min_adaptive_c).detach().numpy().mean()
                                w2_weights = torch.where(w2_weights > min_adaptive_c, w2_weights, min_adaptive_c*torch.ones_like(w2_weights))
                            self.closure.set_w2_weights(w2_weights)

                    payoff_grad_flat = torch._utils._flatten_dense_tensors(payoff_grad_)

                    cg_pack = self.conjugate_gradient(self.closure, payoff_grad_flat,
                                                      stop_criterion=options['wtrpo_opt_params']['stop_criterion'])
                    payoff_inv_dir_flat, payoff_unit_wasssq_gainsq_dist_, payoff_rs_int, payoff_rs_final = cg_pack
                    payoff_inv_dir = torch._utils._unflatten_dense_tensors(payoff_inv_dir_flat, payoff_grad_)

                    check_nan(payoff_inv_dir=payoff_inv_dir)

                    ########################################################################
                    # Determining Trust region maximum coefficient
                    payoff_unit_wasssq_dist_ = torch.sum(payoff_inv_dir_flat.detach() * self.closure(payoff_inv_dir, GSq2WSq_ratio=0., flat_out=True).detach()).detach()

                    payoff_unit_gainsq_dist_ = payoff_unit_wasssq_gainsq_dist_ - payoff_unit_wasssq_dist_
                    u2_ = payoff_unit_wasssq_gainsq_dist_.item()
                    u_ = np.sqrt(u2_ + 1e-20)
                    gTginv_ = tuple_dot(payoff_grad_, payoff_inv_dir, no_grad=True).item()

                    alpha_max = min(1./C_w2, max_w_/u_)

                # Performing line search related stuff.
                if line_search_coeffs is not None:
                    # First, we take a backup of the global rng state for safe keeping
                    vine_rng_state_backup = copy.deepcopy(vine_exploration_policy.np_random.get_state())
                    env_rng_state_backup = copy.deepcopy(self.env.np_random.get_state())

                    # 1) broadcasting rand_seed_init
                    if rank == 0:
                        rand_seed_init = np.random.randint(int(1e9))
                    else:
                        rand_seed_init = None
                    rand_seed_init = comm.bcast(rand_seed_init, root=0)
                    # 2) broadcasting alpha_max
                    if rank > 0:
                        alpha_max = None
                    alpha_max = comm.bcast(alpha_max, root=0)
                    # 3) broadcasting payoff_inv_dir
                    payoff_inv_dir = broadcast_ds(payoff_inv_dir)
                    my_ls_seeds = my_ls_seeds_base + rand_seed_init

                    # Now, let's run the trajectories for each line search coefficient, and collect the payoffs.
                    my_ls_payoffs = []
                    for _, (ls_c_, ls_seed_) in enumerate(zip(my_ls_coeffs, my_ls_seeds)):
                        ls_alpha_ = ls_c_ * alpha_max
                        for p_, backup_p_, payoff_dir_p_ in zip(self.actor.parameters(),
                                                                backup_actor.parameters(),
                                                                payoff_inv_dir):
                            torch.add(backup_p_.data, ls_alpha_, payoff_dir_p_.data, out=p_.data)


                        self.env.np_random.seed(ls_seed_)
                        traj_len_ = options['steps_per_iteration_per_process']
                        gamma_ = options['gamma']

                        if self.env_type == 'stepper':
                            tst_policy = GenericPolicy(vine_exploration_policy, is_greedy=True)
                            traj_payoff = 0.
                            obs_ = self.env.reset()
                            for tt_ in range(traj_len_):
                                action_raw = tst_policy(obs_)
                                action = prepare_action(action_raw, self.env, clip_action=clip_action)
                                obs_, r, done, info = self.env.step(action)
                                traj_payoff = traj_payoff + r * (gamma_**tt_)
                                if done:
                                    break
                        elif self.env_type == 'roller':
                            self.env.set_torch_policy(self.actor)
                            greedy_lite_outdict = self.env.greedy_lite(1, traj_len_, gamma_)
                            traj_payoff = greedy_lite_outdict['eta_greedy'].item()
                        else:
                            raise Exception(f'Unknown env_type {self.env_type}')

                        my_ls_payoffs.append(traj_payoff)

                    # Gathering the results
                    ls_payoffs_full = gather_np(np.array(my_ls_payoffs).reshape(1,-1))
                    if rank == 0:
                        ls_payoffs_full = ls_payoffs_full.reshape(-1)
                        ls_grps = defaultdict(list)
                        for eta_, lsc_ in zip(ls_payoffs_full, lsc_full):
                            ls_grps[lsc_].append(eta_)

                        ls_cfs = sorted(list(ls_grps.keys()))
                        min_ls_cf = min(ls_cfs)


                        ls_etas = []
                        ls_d_etas = []
                        ls_etas_ses = []
                        ls_d_eta_ses = []
                        for _, cf_ in enumerate(ls_cfs):
                            ls_etas.append(np.mean(ls_grps[cf_]))
                            d_eta_ = np.array(ls_grps[cf_]) - np.array(ls_grps[min_ls_cf])
                            ll_ = len(ls_grps[cf_])
                            ls_d_etas.append(np.mean(d_eta_))
                            ls_etas_ses.append(np.std(ls_grps[cf_])/np.sqrt(ll_))
                            ls_d_eta_ses.append(np.std(d_eta_)/np.sqrt(ll_))
                        for cf_, eta_, eta_se_, d_eta_, d_eta_se_ in zip(ls_cfs, ls_etas, ls_etas_ses,
                                                                         ls_d_etas, ls_d_eta_ses):
                            print(f'\t Line search: c=%.3f --> eta=%.3f +/- %.3f,  d_eta =%.3f +/- %.3f' %
                                  (cf_, eta_, eta_se_, d_eta_, d_eta_se_), flush=True)
                        best_cf = ls_cfs[np.argmax(ls_etas)]
                        best_alpha = alpha_max * best_cf

                    # Finally, we return the rng to its original state before the line search
                    self.env.np_random.set_state(env_rng_state_backup)
                    vine_exploration_policy.np_random.set_state(vine_rng_state_backup)
                else:
                    if rank == 0:
                        best_alpha = alpha_max

                if rank == 0:
                    # Computing a bunch of statistics
                    if identity_enabled_:
                        id_stab_loss = tuple_dot(payoff_inv_dir, payoff_inv_dir, no_grad=True).item() * (best_alpha**2)
                        id_term = C_w2 * id_stab_loss * identity_stabilization / 2
                    else:
                        id_term = torch.tensor(0.).double()

                    policy_adv_ = best_alpha * torch.tensor(gTginv_).double()
                    w2_val = torch.tensor(payoff_unit_wasssq_dist_ * (best_alpha ** 2)).double()
                    w2_term = C_w2 * w2_val / 2.
                    gain_term = C_w2 * torch.tensor(payoff_unit_gainsq_dist_ * (best_alpha ** 2)).double() / 2.

                    for name, val in [('w2_term', w2_term), ('gain_term', gain_term), ('w2_val', w2_val),
                                      ('id_term', id_term), ('policy_adv', policy_adv_),
                                      ('payoff_residual_sq_init', payoff_rs_int),
                                      ('payoff_residual_sq_final', payoff_rs_final)]:
                        iter_losses_dict.setdefault(name, [])
                        iter_losses_dict[name].append(val)
                    iter_losses_clip.append(policy_adv_ - w2_term - gain_term - id_term)

                    check_nan(payoff_inv_dir=payoff_inv_dir)
                    for p_, backup_p_, payoff_dir_p_ in zip(self.actor.parameters(),
                                                            backup_actor.parameters(),
                                                            payoff_inv_dir):
                        torch.add(backup_p_.data, best_alpha, payoff_dir_p_.data, out=p_.data)

                    check_nan(net_=self.actor)

                    if rank == 0 and options['verbose']:
                        np.set_printoptions(precision=2)
                        pid_size = get_size(payoff_inv_dir=payoff_inv_dir)['payoff_inv_dir'].item()
                        payoff_size = get_size(payoff_grad_=payoff_grad_)['payoff_grad_'].item()
                        print(f'Rank {rank}:')
                        print(f'\t payoff_unit_wasssq_gainsq_dist_: %.2e' % payoff_unit_wasssq_gainsq_dist_)
                        print(f'\t payoff_unit_wasssq_dist_: %.2e' % payoff_unit_wasssq_dist_)
                        print(f'\t payoff_unit_gainsq_dist_: %.2e' % payoff_unit_gainsq_dist_)
                        print(f'\t payoff_grad_ size: %.5e' % payoff_size)
                        print(f'\t payoff_inv_dir size: %.5e' % pid_size, flush=True)

                    check_nan(net_=self.actor)


                # Final broadcast
                self.actor.broadcast()

            epochs.append(epoch + 1)

            # Critic optimization ##################################################
            iter_losses_V.append(0.)

            # Adaptive Learning rate and exploration params ########################

            if options['adaptive_c_params'] is not None:
                if rank==0:
                    last_w2 = iter_losses_dict['w2_val'][-1]
                    last_advpred = iter_losses_dict['policy_adv'][-1]
                    last_eta = rewards[-1] / correction_ratio

                    sm_factor = options['adaptive_c_params'].get('smoothing_factor', 0.9)
                    if len(rewards) == 1:
                        w2_filtered = [last_w2]
                        advpred_filtered = [last_advpred]
                        eta_filtered = [last_eta, last_eta]
                    else:
                        w2_filtered.append(w2_filtered[-1] * sm_factor + (1.-sm_factor) * last_w2)
                        advpred_filtered.append(advpred_filtered[-1] * sm_factor + (1. - sm_factor) * last_advpred)
                        eta_filtered.append(eta_filtered[-1] * sm_factor + (1. - sm_factor) * last_eta)

                    proposed_c_w2 = (advpred_filtered[-1] - (eta_filtered[-1] - eta_filtered[-2])) / (w2_filtered[-1] + 1e-6)
                    if len(rewards) == 1:
                        prop_c_w2_filtered = [proposed_c_w2]
                    else:
                        prop_c_w2_filtered.append(prop_c_w2_filtered[-1] * sm_factor + (1. - sm_factor) * proposed_c_w2)

                    burnin_iters = options['adaptive_c_params'].get('burnin_iters', 10)
                    if len(rewards) > burnin_iters:
                        safety_mult_ = options['adaptive_c_params'].get('safety_mult_', 1.)
                        C_w2 = max(options['adaptive_c_params'].get('min_c_w2', 1e-2),
                                   prop_c_w2_filtered[-1] * safety_mult_)

                C_w2 = comm.bcast(C_w2, root=0)

            if options['hyperparam_resizer'] is not None:
                if alg_ == 'wppo':
                    if rank == 0:
                        if iter == 0:
                            C_w2_init = C_w2
                            max_w_init = max_w_
                        hyperparam_resizer = options['hyperparam_resizer']
                        cmd_hyperparams = hyperparam_resizer(avg_disc_r=rewards[-1])
                        payoff_grad_coeff = cmd_hyperparams.get('grad', 1.)
                        C_w2 = C_w2_init / cmd_hyperparams.get('C_w2', 1.)
                        max_w_ = max_w_init * cmd_hyperparams.get('max_w', 1.)
                    payoff_grad_coeff = comm.bcast(payoff_grad_coeff, root=0)
                    max_w_ = comm.bcast(max_w_, root=0)
                    if rank == 0:
                        print(f'before update: {C_w2}', flush=True)
                    C_w2 = comm.bcast(C_w2, root=0)
                    if rank == 0:
                        print(f'after update: {C_w2}', flush=True)
                elif alg_ == 'wtrpo':
                    exploration_scale_coeff = -1
                    if rank == 0:
                        if iter == 0:
                            C_w2_init = C_w2
                            max_w_init = max_w_
                        hyperparam_resizer = options['hyperparam_resizer']
                        cmd_hyperparams = hyperparam_resizer(avg_disc_r=rewards[-1])
                        C_w2 = C_w2_init / cmd_hyperparams.get('C_w2', 1.)
                        max_w_ = max_w_init * cmd_hyperparams.get('max_w', 1.)
                        exploration_scale_coeff = cmd_hyperparams.get('exploration_scale', -1)

                    C_w2 = comm.bcast(C_w2, root=0)

                    max_w_ = comm.bcast(max_w_, root=0)
                    exploration_scale_coeff = comm.bcast(exploration_scale_coeff, root=0)
                    if exploration_scale_coeff >= 0:
                        assert hasattr(noise_gen, 'exploration_scale')
                        assert hasattr(noise_gen, 'init_exploration_scale')
                        noise_gen.exploration_scale = noise_gen.init_exploration_scale * exploration_scale_coeff

            if explorer_needs_update:
                if rank == 0:
                    num_grps_ = size // d2_vine

                    e_rows_ = vine_sample_num * (d2_vine-1) * num_grps_
                    e_cols_ = d1_vine * self.env.action_dim

                    msg_  = f'expl_noise.shape = {data_vine["expl_noise"].shape}\n'
                    msg_ += f'(e_rows_, e_cols_) = {(e_rows_, e_cols_)}'
                    assert data_vine['expl_noise'].shape == (e_rows_, e_cols_), msg_

                    expl_noise_batch = data_vine['expl_noise']
                    expl_noise_batch = expl_noise_batch.reshape(-1, d1_vine, self.env.action_dim)
                    A_vine_batch = data_vine['A_vine'].reshape(-1)

                    assert expl_noise_batch.shape[0] == e_rows_
                    assert A_vine_batch.shape[0] == e_rows_

                    explorer.update(expl_noise_batch, A_vine_batch)

                explorer_sdict = explorer.state_dict()
                explorer_sdict = {key : val.detach().cpu() for key, val in explorer_sdict.items()}
                explorer_sdict_bcast = broadcast_ds(explorer_sdict)
                explorer.load_state_dict(explorer_sdict_bcast)

            iter_losses_dict.setdefault('C_w2', [])
            iter_losses_dict['C_w2'].append(C_w2)

            # Post optimization ##################################################
            for name, val_list in iter_losses_dict.items():
                losses_dict.setdefault(name,[])
                losses_dict[name].append(np.mean(val_list))
            losses_clip.append(np.mean(iter_losses_clip))
            losses_V.append(np.mean(iter_losses_V))
            stds.append(np.mean(iter_stds))

            end_time = time.time()
            times_opt.append(end_time - start_time)

            # Output ########################################################
            if rank == 0:
                if options['export_baseline_format']:
                    monitor_dict.setdefault('r', [])
                    monitor_dict.setdefault('t', [])
                    monitor_dict.setdefault('l', [])
                    monitor_dict['r'].append(rewards_dict['undiscounted'][-1])
                    monitor_dict['l'].append(delta_steps)
                    monitor_dict['t'].append(np.sum(times_sample)+np.sum(times_opt))
                    if time.time() > (monitor_file_last_mod_time + 60.):
                        try:
                            first_line = '# {"t_start": '+str(time.time())+', "env_id": "Pendulum-v0"}'
                            with open(monitor_file, 'w') as csvfile:
                                csvfile.write(first_line + "\n")
                                monitor_writer = csv.writer(csvfile, dialect='excel')#, quoting=csv.QUOTE_NONNUMERIC)
                                cols = ['r', 'l', 't']
                                num_rows = len(monitor_dict[cols[0]])
                                monitor_writer.writerow(cols)
                                for row_i in range(num_rows):
                                    monitor_writer.writerow([monitor_dict[col][row_i] for _, col in enumerate(cols)])
                        except IOError as exc_:
                            print("I/O error: {0}".format(exc_))
                        monitor_file_last_mod_time = time.time()

            if rank == 0 and options['enable_tb']:
                writer.add_scalar('reward', rewards[-1], n_steps)
                writer.add_scalar('loss/clip', losses_clip[-1], n_steps)
                if not alg_is_greedy_:
                    writer.add_scalar('loss/V', losses_V[-1], n_steps)
                    writer.add_scalar('std', stds[-1], n_steps)
                writer.add_scalar('time/sample', times_sample[-1], n_steps)
                writer.add_scalar('time/opt', times_opt[-1], n_steps)
                writer.add_scalar('loss/epochs', epochs[-1], n_steps)
                for name, val in losses_dict.items():
                    writer.add_scalar(f'loss/{name}', val[-1], n_steps)

                if self.actor_class == 'rbf':
                    rbf_stds = np.sqrt(np.mean(np.exp(self.actor.log_rbf_stds.detach().numpy())**2, axis=0))
                    for i_ in range(len(rbf_stds)):
                        writer.add_scalar(f'stats/rbf_std_{i_}', rbf_stds[i_], n_steps)

                if options['wtrpo_opt_params'].get('min_adaptive_c', None) is not None:
                    writer.add_scalar('loss/small_c_ratio', small_c_ratio, n_steps)

                if line_search_coeffs is not None:
                    writer.add_scalar('loss/line_search_coeff', best_cf, n_steps)

                if explorer_needs_update:
                    for key, val_tensor in explorer_sdict.items():
                        val_np = val_tensor.detach().cpu().numpy().reshape(-1)
                        for ii, vv in enumerate(val_np):
                            writer.add_scalar(f'exploration/{key}_{ii}', vv, n_steps)

                if options['output_histograms']:
                    writer.add_histogram('advantage_hist', data_vine['A_vine'], n_steps)

                if steps_since_last_snapshot >= options['steps_per_snapshot']:
                    steps_since_last_snapshot = 0
                    snapshot_tail = f'snapshot_{n_steps}'
                    snapshot_relative = os.path.join(logdir_tail, snapshot_tail)
                    snapshot_dir = os.path.join(logdir, snapshot_tail)
                    os.makedirs(snapshot_dir, exist_ok=True)

                    # save actor and critic
                    save_prefix = os.path.join(snapshot_dir, 'agent')
                    self.save(save_prefix)

                    # link logdir/latest to the snapshot_dir
                    link_name = os.path.join(options['logdir_base'], 'latest')
                    try:
                        os.unlink(link_name)
                        os.symlink(snapshot_relative, link_name)
                    except OSError:
                        pass
                    except FileExistsError:
                        pass

                    # link logdir/<logdir_tail>/latest to the snapshot_dir
                    link_name = os.path.join(logdir, 'latest')
                    try:
                        os.unlink(link_name)
                        os.symlink(snapshot_tail, link_name)
                    except OSError:
                        pass
                    except FileExistsError:
                        pass

                    # trajectory line plot
                    if options['greedy_plot']:
                        plot_env = options['ploting_environment']
                        if plot_env is None:
                            plot_env = self.env
                        vine_exploration_policy.set_d1(d1_vine)
                        do_backup_env_rng = (options['ploting_environment'] is not None)
                        vine_rng_state_plt = copy.deepcopy(vine_exploration_policy.np_random.get_state())
                        if do_backup_env_rng:
                            env_rng_state_plt = copy.deepcopy(plot_env.np_random.get_state())
                        temp_t = -1
                        plot_policy = GenericPolicy(vine_exploration_policy)
                        plt.ioff()

                        if not hasattr(plot_env, 'simulate_and_plot'):
                            obs_plot_lists = options.get('obs_plot_lists', None)
                            act_plot_lists = options.get('act_plot_lists', None)
                            fig = simulate_and_plot(plot_env, plot_policy, obs_plot_lists, act_plot_lists,
                                                    traj_len=options['steps_per_iteration_per_process'],
                                                    gamma=options['gamma'], clip_action=clip_action)
                        else:
                            fig = plot_env.simulate_and_plot(plot_policy)
                        writer.add_figure('simulation_greedy', fig, n_steps)
                        plt.ion()
                        vine_exploration_policy.np_random.set_state(vine_rng_state_plt)
                        if do_backup_env_rng:
                            plot_env.np_random.set_state(env_rng_state_plt)
                    if options['noisy_plot']:
                        assert False, 'Why would you ever need this?!'
                        plt.ioff()
                        fig = self.env.simulate_and_plot(self.action)
                        writer.add_figure('simulation_noisy', fig, n_steps)
                        plt.ion()

        out_dict = {'rewards': rewards,
                    'undiscounted_rewards':rewards_dict['undiscounted'],
                    'losses_clip': losses_clip,
                    'losses_V': losses_V,
                    'stds': stds,
                    'times_sample': times_sample,
                    'times_opt': times_opt,
                    'epochs': epochs
                    }
        if rank == 0:
            print('I am done!', flush=True)

        if options['out_file'] is not None and rank==0:
            if options['out_file'].endswith('npz'):
                out_dict_np = {x:np.array(y) for x,y in out_dict.items()}
                np.savez(options['out_file'], **out_dict_np)
            elif options['out_file'].endswith('json'):
                fp = open(options['out_file'], 'w')
                mystr = json.dumps(out_dict)
                fp.write(mystr)
                fp.flush()
                fp.close()

        return out_dict

    def gainsq_grad_finite_diff(self, s, eps_finite_diff=1e-12, policy=None, **kwargs):
        policy = policy or self.actor
        mu, std = policy(s)

        #obs_dim = self.env.observation_dim
        obs_dim = s.shape[1]

        act_dim = self.env.action_dim
        out_type = kwargs.get('out_type', 'flat')

        assert out_type in ['flat', 'grad.data', 'seq']
        assert len(mu.shape) == 2 and len(s.shape) == 2

        obs_eye_mat = torch.eye(obs_dim, dtype=torch.double)

        s_need_grad = s.requires_grad
        bs = s.shape[0]
        s.detach_()

        full_gain = 0.
        for i in range(obs_dim):
            s_new = s + obs_eye_mat[i] * eps_finite_diff
            mu_new, std_new = policy(s_new)
            full_gain += torch.sum((mu_new - mu) ** 2) / bs
        # full_gain = full_gain.sqrt()/eps_finite_diff
        full_gain = full_gain / (eps_finite_diff ** 2)
        if out_type in ('flat', 'seq'):
            policy_params = tuple(self.actor.parameters())
            nabla_th_gain_seq = torch.autograd.grad(full_gain, policy_params, grad_outputs=None, create_graph=False,
                                                    retain_graph=True, only_inputs=True, allow_unused=True)
            nabla_th_gain_seq = tuple(
                x if x is not None else torch.zeros_like(y) for x, y in zip(nabla_th_gain_seq, policy_params))

            s.requires_grad_(s_need_grad)
            if out_type == 'flat':
                nabla_th_gain = torch._utils._flatten_dense_tensors(nabla_th_gain_seq)
                return nabla_th_gain, full_gain
            else:
                return nabla_th_gain_seq, full_gain

        elif out_type == 'grad.data':
            full_gain.backward()
            s.requires_grad_(s_need_grad)
            return None, full_gain

    def conjugate_gradient(self, closure, b, x=None, stop_criterion=('eps', 1e-8), verbose=False):
        ref_eps_sq = 1e-20
        if x is None:
            x = torch.zeros_like(b)
        with torch.no_grad():
            if torch.sum(b ** 2) < ref_eps_sq:
                return x, torch.tensor(0.).double(), 0., 0.
        r = torch.zeros_like(b)
        r = closure(x, out=r).detach()
        r.sub_(b)
        r.neg_()

        p = r.clone().detach()
        rsold = torch.dot(r, r)
        rs_init = rsold.item()

        dim_ = torch.numel(b)
        Ap = torch.zeros_like(p)
        for k in range(dim_):
            Ap = closure(p, out=Ap).detach()
            ptAp = torch.dot(Ap, p)
            alpha = rsold / ptAp
            x.add_(alpha, p)
            r.sub_(alpha, Ap)

            rsnew = torch.dot(r, r)
            if verbose:
                print(f'k: {k}, r**2: {rsnew}', flush=True)

            finished=False
            for cri_cntr in range(len(stop_criterion)//2):
                cond = stop_criterion[2*cri_cntr]
                argcond = stop_criterion[2*cri_cntr + 1]
                if cond == 'eps':
                    finished = finished or rsnew < argcond ** 2
                elif cond == 'iter':
                    finished = finished or k >= argcond
                elif cond == 'frac':
                    finished = finished or rsnew < rs_init * argcond
                else:
                    raise Exception(f'Unkown stop_criterion: {stop_criterion}')

            if rsold < ref_eps_sq:
                print('CG Residual is too small. Consider using a smaller number of iterations or a better stopping criterion.', flush=True)

            if finished:
                xtAx = torch.dot(b, x) - torch.dot(r, x)
                return x, xtAx, rs_init, rsnew.item()

            p.mul_(rsnew / rsold).add_(r)
            rsold = rsnew
        xtAx = torch.dot(b, x) - torch.dot(r, x)
        return x, xtAx, rs_init, rsnew.item()

    def regenerate_monitor(self, logdir_bl, env_name, clip_action=None):
        all_dirs = os.listdir(logdir_bl)
        snap_dirs = [x for x in all_dirs if x.startswith('snapshot')]
        snap_dirs = sorted(snap_dirs, key=lambda x: int(x.split('_')[-1]))
        snap_steps = [int(x.split('_')[-1]) for x in snap_dirs]
        snap_times = [os.stat(os.path.join(logdir_bl, x)).st_mtime for x in snap_dirs]
        first_snap_time = snap_times[0]
        snap_times = (np.array(snap_times) - first_snap_time).tolist()

        possible_monitor_file = os.path.join(logdir_bl, '0.0.monitor.csv')
        monitor_exists = os.path.exists(possible_monitor_file)
        if monitor_exists:
            lines = [line.rstrip('\n') for line in open(possible_monitor_file, 'r')]
            prop_dict = json.loads(lines[0].split('#')[-1])
            old_monitor = np.genfromtxt(lines[2:], dtype=None, names=lines[1], delimiter=",")
            cum_steps = np.cumsum(old_monitor['l'])
            final_times = []
            for snap_step in snap_steps:
                monit_idx = np.where(cum_steps == snap_step)[0][0]
                final_times.append(old_monitor['t'][monit_idx])
            final_times = np.array(final_times)
        else:
            final_times = snap_times
        final_lengths = [int(snap_steps[0])]
        final_lengths = final_lengths + [int(snap_steps[i] - snap_steps[i - 1]) for i in range(1, len(snap_steps))]

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        my_rewards = []
        my_steps = []
        out_rows = np.ceil(float(len(snap_steps)) / size)
        for ord_num, snap_step in enumerate(snap_steps):
            if (ord_num % size) == rank:
                self.load(logdir_bl+'/snapshot_' + str(int(snap_step)) + '/agent')
                assert self.env_type == 'stepper', 'too old for roller envs.'
                with torch.no_grad():
                    r = []
                    s_next = self.env.reset()
                    while True:
                        (mu_t, std_t) = self.actor(torch.from_numpy(s_next).double())
                        assert clip_action is not None, 'Make sure you specify clip_action. Not implemented before...'
                        (s_next, r_t, done, _) = self.env.step(self.prepare_action(mu_t.detach().numpy(), self.env, clip_action=clip_action))
                        r.append(r_t)
                        if done:
                            break
                    undiscounted_payoff = np.sum(r)
                    my_rewards.append(undiscounted_payoff)
                    my_steps.append(snap_step)

        while len(my_rewards) < out_rows:
            my_rewards.append(0.)
            my_steps.append(-1.)

        local_array = [my_steps, my_rewards]
        local_array = np.concatenate([np.array(my_steps).reshape(-1,1), np.array(my_rewards).reshape(-1,1)], axis=1)
        local_array = np.copy(local_array.astype(np.float))
        assert len(local_array.shape) == 2
        num_rows = local_array.shape[0]
        num_cols = local_array.shape[1]
        global_array = None
        if rank == 0:
            global_array = np.empty((num_rows * size, num_cols), dtype=local_array.dtype)
        comm.Gather(local_array, global_array, root=0)
        if rank == 0:
            global_array = global_array[global_array[:, 0] > 0]
            all_steps = global_array[:, 0]
            all_rewards = global_array[:, 1]
            ord_steps = np.argsort(all_steps)
            all_steps = all_steps[ord_steps]
            all_rewards = all_rewards[ord_steps]
            assert np.all(all_steps == np.array(snap_steps)), (global_array,  snap_steps)

            out_lines = ['# {"t_start": 0.0, "env_id": "' + env_name + '"}']
            out_lines.append('r,l,t')
            for i in range(len(snap_steps)):
                out_lines.append(str(all_rewards[i]) + ',' + str(final_lengths[i]) + ',' + str(final_times[i]))

            if monitor_exists:
                os.rename(possible_monitor_file, os.path.join(logdir_bl, 'oldmonitorcsv'))
            with open(possible_monitor_file, 'w') as filehandle:
                filehandle.writelines("%s\n" % line_ for line_ in out_lines)

def extend_nn_input(init_net, new_input_size, new2old_input_mapping):
    """
        This function takes a nn module that has a first linear layer, and extends the input
        by adding zero weighted units at the input layer. It also allows specifying an input mapping,
        which could also help rearrange the input order.

        Input Arguments:
            init_net(nn.Module): Initial neural net which you want to extend its input dimensions
            new_input_size(int): This is the desired new input size
            new2old_input_mapping(dict): A dictionary in the format of {new_index : old_index}

        Output Arguments:
            out_net(nn.Module): Output neural net that is supposed to give the same output for
                                vectors with right mapping.
    """
    net2 = copy.deepcopy(init_net)
    first_layer = list(net2.children())[0]
    assert isinstance(first_layer, torch.nn.Linear), 'First layer should be a linear layer'
    old_in_size = first_layer.in_features
    old_out_size = first_layer.out_features
    use_bias = not (first_layer.bias is None)

    new_first_layer = torch.nn.Linear(new_input_size, old_out_size, bias=use_bias)
    with torch.no_grad():
        new_first_layer.weight.data.zero_()
        if new_first_layer.bias is not None:
            new_first_layer.bias.data = first_layer.bias.data

        for new_idx, old_idx in new2old_input_mapping.items():
            if old_idx >= 0:
                new_first_layer.weight.data[:, new_idx] = first_layer.weight.data[:, old_idx]
    out_net = torch.nn.Sequential(new_first_layer, *list(net2.children())[1:])
    return out_net


def test_extend_nn_input():
    init_net = torch.nn.Sequential(torch.nn.Linear(2, 120), torch.nn.ReLU(), torch.nn.Linear(120, 84), torch.nn.ReLU(),
                                   torch.nn.Linear(84, 3))
    new2old_input_mapping = {0: -1, 1: 0, 2: 1}
    new_input_size = 3

    extended_net = extend_nn_input(init_net, new_input_size, new2old_input_mapping)

    x1 = torch.tensor([[1.3, 2.1]])
    x2 = torch.tensor([[6.1, 1.3, 2.1]])

    y1 = init_net(x1)
    y2 = extended_net(x2)

    print('The error after extension: ', (y1 - y2).detach().cpu().numpy())


def check_nan(**kwargs):
    for gen_name, _net in kwargs.items():
        if isinstance(_net, torch.nn.Module):
            for p_name_, p_ in _net.named_parameters():
                assert not (np.isnan(p_.detach().numpy()).any()), f'gen_name: {gen_name}, p_name_:{p_name_}, p_:{p_}'
        elif isinstance(_net, tuple):
            for idx_, p_ in enumerate(_net):
                assert not (np.isnan(p_.detach().numpy()).any()), f'gen_name: {gen_name}, idx_:{idx_}, p_:{p_}'
        else:
            raise Exception(f'Unknown input: {gen_name}, {_net}')


def get_size(**kwargs):
    out_dict = dict()
    for gen_name, _net in kwargs.items():
        if isinstance(_net, torch.nn.Module):
            size_sq = 0.
            for p_name_, p_ in _net.named_parameters():
                size_sq += torch.sum(p_ ** 2)
            out_dict[gen_name] = size_sq.sqrt()
        elif isinstance(_net, tuple):
            size_sq = 0.
            for idx_, p_ in enumerate(_net):
                size_sq += torch.sum(p_ ** 2)
            out_dict[gen_name] = size_sq.sqrt()
        else:
            raise Exception(f'Unknown input: {gen_name}, {_net}')

    return out_dict


def project_to_bounds_(x_, bounds_):
    out_ = []
    for idx_, (a_, (lb_, up_)) in enumerate(zip(x_, bounds_)):
        assert not (np.isnan(a_))
        if a_ > up_:
            out_.append(up_)
        elif a_ < lb_:
            out_.append(lb_)
        else:
            out_.append(a_)
    return tuple(out_)


def mycumprodsum(my_delta, my_gamma):
    if torch.is_tensor(my_delta):
        my_delta = my_delta.to(dtype=torch.float)
        c = torch.arange(my_delta.numel()).to(my_delta)
        c = np.power(1. / my_gamma, c)
        a = my_delta * c
        a = torch.cumsum(a, dim=0)
        return a / c
    else:
        my_delta = np.array(my_delta)
        c = np.arange(my_delta.size)
        c = np.power(1. / my_gamma, c)
        a = my_delta * c
        a = np.cumsum(a)
        return a / c


def mycumprodsum_rev(my_delta, my_gamma):
    if torch.is_tensor(my_delta):
        return mycumprodsum(my_delta.flip(0), my_gamma).flip(0)
    else:
        return mycumprodsum(my_delta[::-1], my_gamma)[::-1]


def tuple_dot(a, b, no_grad=False):
    ab_dotp = 0.
    for x_, y_ in zip(a, b):
        x_1d = x_.reshape(-1)
        y_1d = y_.reshape(-1)
        if no_grad:
            x_1d = x_1d.detach()
            y_1d = y_1d.detach()
        ab_dotp += torch.dot(x_1d, y_1d)
    return ab_dotp


def qp_interior_maximize(x_lite, y, bounds, max_func_val=None):
    M = np.zeros((6, 6))
    M[:, 0] = 1.
    M[:, 1] = x_lite[:, 0]
    M[:, 2] = x_lite[:, 1]
    M[:, 3] = (x_lite[:, 0] ** 2) / 2
    M[:, 4] = (x_lite[:, 1] ** 2) / 2
    M[:, 5] = x_lite[:, 0] * x_lite[:, 1]

    x = np.linalg.solve(M, y)
    f, f_x, f_y, f_xx, f_yy, f_xy = tuple(x)

    H = np.array([[f_xx, f_xy],
                  [f_xy, f_yy]])
    g = np.array([[f_x],
                  [f_y]])

    constraints_mat = np.array([[1., 0.],
                                [0., 1.],
                                [-1., 0.],
                                [0., -1.]])

    opt_dxdy = np.linalg.inv(H) @ (-1. * g)
    opt_dx, opt_dy = tuple(opt_dxdy.reshape(-1))

    x_lb = min(bounds[0])
    x_ub = max(bounds[0])
    y_lb = min(bounds[1])
    y_ub = max(bounds[1])

    def find_opt_lincomb(x1, x2):
        x1 = np.array(x1).reshape(2, 1)
        x2 = np.array(x2).reshape(2, 1)
        a = (x1.T @ H @ x1).item()
        b = (x2.T @ H @ x2).item()
        c = (x1.T @ H @ x2).item()
        d = (g.T @ (x1 - x2)).item()

        h = (a + b - 2 * c)
        m = (c - b + d)
        if h == 0:
            alpha_star = 0.  # doesn't really matter!
        else:
            alpha_star = -m / h

        return alpha_star * x1 + (1 - alpha_star) * x2

    left_down = [x_lb, y_lb]
    right_down = [x_ub, y_lb]
    left_up = [x_lb, y_ub]
    right_up = [x_ub, y_ub]

    cand1 = find_opt_lincomb(left_down, right_down)
    cand2 = find_opt_lincomb(left_down, left_up)
    cand3 = find_opt_lincomb(left_up, right_up)
    cand4 = find_opt_lincomb(left_down, right_down)

    def check_valid(x_):
        x_ = np.array(x_).reshape(-1)
        return x_lb <= x_[0] <= x_ub and y_lb <= x_[1] <= y_ub

    valid_cands = [x_ for x_ in [opt_dxdy, cand1, cand2, cand3, cand4, left_down, right_down, left_up, right_up] if
                   check_valid(x_)]

    def func_(x_):
        x_ = np.array(x_).reshape(2, 1)
        return (g.T @ x_).item() + (x_.T @ H @ x_).item() / 2

    valid_cand_vals = [func_(x_) for x_ in valid_cands]
    final_res = tuple(np.array(valid_cands[np.argmax(valid_cand_vals)]).reshape(-1))
    final_val = np.max(valid_cand_vals)
    # assert final_val >= 0., f'final_val:{final_val}'
    if max_func_val is not None:
        if final_val >= max_func_val:
            x_ = np.array(final_res).reshape(2, 1)
            b = (g.T @ x_).item()
            a = (x_.T @ H @ x_).item() / 2
            c = -1 * max_func_val
            alpha = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)

            final_res = tuple((alpha * x_).reshape(-1))
            final_val = func_(final_res)

            # assert final_val<=max_func_val, f'max_func_val:{max_func_val}, final_val:{final_val}'

    return final_res, final_val

def prepare_action(action, env, clip_action=False):
    # Unfortunately, there is some sort of a sneakiness in the gym environments, where if you don't
    # clip your actions before passing them to the environemnt, you may get unnecessary negative rewards
    # and the this unnecessary negative reward has a devastating reward shaping effect.

    # This is quite seeable in the walker environment...
    #assert np.all(env.action_space.high == 1.), env.action_space.high
    if clip_action:
        return np.clip(action, env.action_space.low, env.action_space.high)
    else:
        return action

def simulate_and_plot(env, policy, obs_plot_lists=None, act_plot_lists=None, reward_keys=None, traj_len=200,
                      fig_width=6, fig_height_per_plot=2, gamma=1., title_postfix='', plot_fft=False, clip_action=False):
    # Ex:
    # obs_plot_lists = [[(0, 'x'), (1, 'y')], [(2,'z')]]
    # act_plot_lists = [[(0, 'a'), (1, 'b')]]

    traj = dict()
    t_sim = list()
    obs_plot_lists = obs_plot_lists or []
    act_plot_lists = act_plot_lists or []

    obs = env.reset()
    info_keys = None
    for t in range(traj_len):
        action = policy(obs)
        action = prepare_action(action, env, clip_action=clip_action)
        obs, r, done, info = env.step(action)

        info_keys = info_keys or list(info.keys())
        for key, val in info.items():
            traj.setdefault(key, [])
            traj[key].append(val)

        traj.setdefault('total_reward', [])
        traj['total_reward'].append(r)

        for obs_plot_list in obs_plot_lists:
            for i, key in obs_plot_list:
                traj.setdefault(key, [])
                traj[key].append(obs[i])

        for act_plot_list in act_plot_lists:
            for i, key in act_plot_list:
                traj.setdefault(key, [])
                traj[key].append(action[i])

        t_sim.append(t)
        if done:
            break

    traj_okay = {}
    for key in traj.keys():
        if len(t_sim) == len(traj[key]):
            traj_okay[key] = traj[key]
    traj = traj_okay
    info_keys = [ik_ for ik_ in info_keys if ik_ in traj.keys()]

    reward_keys = reward_keys or [x for x in info_keys if 'reward' in x]
    non_reward_keys = [x for x in info_keys if 'reward' not in x]
    n_plots = 2 + 2 * int(len(reward_keys) > 0) + len(obs_plot_lists) + len(act_plot_lists) + int(
        len(non_reward_keys) > 0)
    n_cols = 1 + int(plot_fft)
    fig, axes = plt.subplots(n_plots, n_cols, figsize=(n_cols * fig_width, n_plots * fig_height_per_plot), sharex='col')
    axes = np.array(axes).reshape(n_plots, n_cols)
    axes[0, 0].set_title(f'average reward per step = {np.mean(traj["total_reward"]):.4}{title_postfix}')

    i_axis = -1
    if len(reward_keys):
        i_axis += 1
        ax = axes[i_axis, 0]
        for key in reward_keys:
            disc_r = traj[key] * (gamma**np.array(t_sim))
            ax.plot(t_sim, disc_r, linewidth=1, label=key.replace('reward', 'r'))
        ax.legend(bbox_to_anchor=(0., 1.2, 1., .2), loc='lower left',
                  ncol=3, mode="expand", borderaxespad=0.)
        ax.grid()
        ax.set_yscale('symlog')

    i_axis += 1
    ax = axes[i_axis, 0]
    ax.plot(t_sim, traj['total_reward'], 'k-', linewidth=1, label='reward')
    ax.legend(loc='upper right')
    ax.grid()
    ax.set_yscale('symlog')

    # FIXME: scaled wrong
    if len(reward_keys):
        i_axis += 1
        ax = axes[i_axis, 0]
        for key in reward_keys:
            disc_r = traj[key] * (gamma ** np.array(t_sim))
            ax.plot(t_sim, np.cumsum(disc_r), linewidth=1, label=f"cumul {key.replace('reward', 'r')}")
        ax.grid()
        ax.set_yscale('symlog')

        if plot_fft:
            # This has nothing to do with the fft!
            ax_ft = axes[i_axis, 1]
            for key in reward_keys:
                disc_r_final = np.sum(traj[key] * (gamma ** np.array(t_sim)))
                f_sim = np.fft.fftfreq(len(traj[key]))
                ax_ft.step(f_sim, [disc_r_final for _ in f_sim], linewidth=1, label=key+'_cum_final')
            ax_ft.legend(loc='upper right')
            ax_ft.grid()

    # FIXME: scaled wrong
    i_axis += 1
    ax = axes[i_axis, 0]
    ax.plot(t_sim, np.cumsum(traj['total_reward']), 'k-', linewidth=1, label='cumulative reward')
    ax.legend(loc='upper right')
    ax.grid()
    ax.set_yscale('symlog')

    for obs_plot_list in obs_plot_lists:
        i_axis += 1
        ax = axes[i_axis, 0]
        for i, key in obs_plot_list:
            ax.step(t_sim, traj[key], linewidth=1, label=key)
        ax.legend(loc='upper right')
        ax.grid()

        if plot_fft:
            ax_ft = axes[i_axis, 1]
            for i, key in obs_plot_list:
                f_sim, sig_ft = fft_proc(traj[key])
                ax_ft.step(f_sim, sig_ft, linewidth=1, label=key+'_ft')
            ax_ft.legend(loc='upper right')
            ax_ft.grid()

    for act_plot_list in act_plot_lists:
        i_axis += 1
        ax = axes[i_axis, 0]
        for i, key in act_plot_list:
            ax.step(t_sim, traj[key], linewidth=1, label=key)
        ax.legend(loc='upper right')
        ax.grid()

        if plot_fft:
            ax_ft = axes[i_axis, 1]
            for i, key in act_plot_list:
                f_sim, sig_ft = fft_proc(traj[key])
                ax_ft.step(f_sim, sig_ft, linewidth=1, label=key+'_ft')
            ax_ft.legend(loc='upper right')
            ax_ft.grid()

    if len(non_reward_keys):
        i_axis += 1
        ax = axes[i_axis, 0]
        for key in non_reward_keys:
            ax.step(t_sim, traj[key], linewidth=1, label=key)
        ax.legend(loc='upper right')
        ax.grid()

        if plot_fft:
            ax_ft = axes[i_axis, 1]
            for key in non_reward_keys:
                f_sim, sig_ft = fft_proc(traj[key])
                ax_ft.step(f_sim, sig_ft, linewidth=1, label=key+'_ft')
            ax_ft.legend(loc='upper right')
            ax_ft.grid()

    ax.set_xlabel('time (s)')
    if plot_fft:
        ax_ft.set_xlabel('freq (hz)')

    fig.set_tight_layout(dict(h_pad=0.1))

    return fig

def fft_proc(in_s):
    s = np.array(in_s)
    freq = np.fft.fftfreq(s.shape[0])
    s_fft = np.fft.fft(s)
    sp = np.abs(s_fft)
    freq_tup, sp_tup = zip(*sorted(zip(freq.tolist(), sp.tolist())))
    freq, sp = np.array(list(freq_tup)), np.array(list(sp_tup))
    return freq, sp

def sec_order_eq_solver(a_,b_,c_):
    delta_ = b_**2 - 4. * a_ * c_
    if a_ == 0:
        return None, None
    elif delta_ > 0.:
        return (-1. * b_ - np.sqrt(delta_)) / (2. * a_), (-1. * b_ + np.sqrt(delta_)) / (2. * a_)
    else:
        return None, None
