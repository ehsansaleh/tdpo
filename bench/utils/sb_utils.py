import numpy as np
import tensorflow as tf
import multiprocessing
import gym
import time
from collections import OrderedDict, defaultdict
import os, sys

disable_mpi4py = 'NOMPI4PY' in os.environ
if disable_mpi4py:
    sys.modules['mpi4py'] = None
    # This will prevent stable_baselines from
    # importing mpi4py and any relavant methods

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines.common.vec_env.base_vec_env import VecEnv, CloudpickleWrapper
from stable_baselines.common.distributions import make_proba_dist_type, CategoricalProbabilityDistribution, \
    MultiCategoricalProbabilityDistribution, DiagGaussianProbabilityDistribution, BernoulliProbabilityDistribution
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.td3.policies import FeedForwardPolicy as TD3FeedForwardPolicy
from stable_baselines.sac.policies import mlp as mlpmaker

###############################################################################
###################################### ppo2 ###################################
###############################################################################

class BetterMlpPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super(BetterMlpPolicy, self).__init__(*args, **kwargs)


    def _setup_init(self):
        """Sets up the distributions, actions, and value."""
        with tf.variable_scope("input", reuse=False):
            self._action_ph = tf.placeholder(dtype=self.ac_space.dtype, shape=(self.n_batch,) + self.ac_space.shape,
                                             name="action_ph")
        with tf.variable_scope("output", reuse=True):
            assert self.policy is not None and self.proba_distribution is not None and self.value_fn is not None
            self._action = self.proba_distribution.sample()
            self._deterministic_action = self.proba_distribution.mode()
            self._neglogp = self.proba_distribution.neglogp(self.action)
            # This is the part that we had to add: an action placeholder for
            # evaluating the log-likelihood of some **given** actions.
            if self.action_ph is not None:
                action_f32 = tf.cast(self.action_ph, tf.float32)
                self.neglogp_action_ph = self.proba_distribution.neglogp(action_f32)
            if isinstance(self.proba_distribution, CategoricalProbabilityDistribution):
                self._policy_proba = tf.nn.softmax(self.policy)
            elif isinstance(self.proba_distribution, DiagGaussianProbabilityDistribution):
                self._policy_proba = [self.proba_distribution.mean, self.proba_distribution.std]
            elif isinstance(self.proba_distribution, BernoulliProbabilityDistribution):
                self._policy_proba = tf.nn.sigmoid(self.policy)
            elif isinstance(self.proba_distribution, MultiCategoricalProbabilityDistribution):
                self._policy_proba = [tf.nn.softmax(categorical.flatparam())
                                     for categorical in self.proba_distribution.categoricals]
            else:
                self._policy_proba = []  # it will return nothing, as it is not implemented
            self._value_flat = self.value_fn[:, 0]

def traj_seg_roller(actor_weights_np, env, horizon):
    episode_start = True
    ep_ret = 0
    ep_len = 0
    env.reset()
    while True:
        ep_rets = []
        ep_lens = []
        steps_taken = 0
        partial_outs = defaultdict(list)
        while steps_taken < horizon:
            n_steps = horizon - steps_taken
            d = env.multiple_steps(n_steps, actor_weights_np, expl_noise=None, policy_lib='np')
            for key, val in d.items():
                partial_outs[key].append(val)
            last_dones = d['dones']
            last_rewards = d['rewards']
            steps_taken = steps_taken + last_dones.size
            ep_ret = ep_ret + last_rewards.sum()
            ep_len = ep_len + last_dones.size
            if last_dones[-1]:
                ep_rets.append(ep_ret)
                ep_ret = 0
                ep_lens.append(ep_len)
                ep_len = 0
                env.reset()

        agg_dict = dict()

        ################# s/a/r/d ###################
        for key, val in partial_outs.items():
            agg_dict[key] = np.concatenate(val, axis=0)

        observations = agg_dict['observations']
        actions = agg_dict['actions']
        rewards = agg_dict['rewards']
        dones = agg_dict['dones']

        ################# vpreds ####################
        #vpreds = policy.value(observations)

        ############### ep_starts ###################
        episode_starts = np.empty_like(dones)
        episode_starts[0] = episode_start
        episode_starts[1:] = dones[:-1]
        episode_start = dones[-1]

        ############### nextvpred ###################
        #nextvpred = vpreds[-1] * (1- episode_start)

        ############# total_timestep ################
        total_timestep = dones.size

        yield {"observations": observations,
               "rewards": rewards,
               "dones": dones,
               "episode_starts": episode_starts,
               "episode_start": episode_start,
               "true_rewards": rewards,
               #"vpred": vpreds,
               "actions": actions,
               #"nextvpred": nextvpred,
               "ep_rets": ep_rets,
               "ep_lens": ep_lens,
               "ep_true_rets": ep_rets,
               "total_timestep": total_timestep,
              }

def get_tf_actor_weights(tf_session, naming='trpo'):
    tf_graph = tf_session.graph
    if naming in ('trpo', 'ppo', 'ppo1', 'ppo2'):
        fc1_w_tensor = tf_graph.get_tensor_by_name("model/pi_fc0/w:0")
        fc2_w_tensor = tf_graph.get_tensor_by_name("model/pi_fc1/w:0")
        fc3_w_tensor = tf_graph.get_tensor_by_name("model/pi/w:0")
        fc1_b_tensor = tf_graph.get_tensor_by_name("model/pi_fc0/b:0")
        fc2_b_tensor = tf_graph.get_tensor_by_name("model/pi_fc1/b:0")
        fc3_b_tensor = tf_graph.get_tensor_by_name("model/pi/b:0")
        logstd_tensor = tf_graph.get_tensor_by_name("model/pi/logstd:0")

        pkg = tf_session.run([fc1_w_tensor, fc2_w_tensor, fc3_w_tensor,
                              fc1_b_tensor, fc2_b_tensor, fc3_b_tensor,
                              logstd_tensor])
        fc1_w, fc2_w, fc3_w, fc1_b, fc2_b, fc3_b, logstd = pkg
        policy_logstd = logstd
    elif naming in ('td3',):
        fc1_w_tensor = tf_graph.get_tensor_by_name("model/pi/fc0/kernel:0")
        fc2_w_tensor = tf_graph.get_tensor_by_name("model/pi/fc1/kernel:0")
        fc3_w_tensor = tf_graph.get_tensor_by_name("model/pi/dense/kernel:0")
        fc1_b_tensor = tf_graph.get_tensor_by_name("model/pi/fc0/bias:0")
        fc2_b_tensor = tf_graph.get_tensor_by_name("model/pi/fc1/bias:0")
        fc3_b_tensor = tf_graph.get_tensor_by_name("model/pi/dense/bias:0")

        pkg = tf_session.run([fc1_w_tensor, fc2_w_tensor, fc3_w_tensor,
                              fc1_b_tensor, fc2_b_tensor, fc3_b_tensor])
        fc1_w, fc2_w, fc3_w, fc1_b, fc2_b, fc3_b = pkg
        policy_logstd = None
    else:
        raise ValueError(f'Unknown naming {naming}')

    named_parameters = dict(fc1 = fc1_w.T,
                            fc2 = fc2_w.T,
                            fc3 = fc3_w.T,
                            fc1_bias = fc1_b,
                            fc2_bias = fc2_b,
                            fc3_bias = fc3_b,
                            policy_logstd = policy_logstd)

    return named_parameters

def mycumprodsum(my_delta, my_gamma):
    my_delta = np.array(my_delta)
    c = np.arange(my_delta.size)
    c = np.power(1./my_gamma, c)
    a = my_delta * c
    a = np.cumsum(a)
    return a / c

def mycumprodsum_numstable(my_delta, my_gamma):
    """
    The mycumprodsum above computes "1/gamma**T" in the
    c = np.power(1./my_gamma, c) line. This number can easily overflow
    if 1/gamma was big enough. This function mitigates this problem by
    dividing my_delta in binary chunks recursively and then stitching
    the outputs carefully together. When everything is numerically stable,
    this function should return identical outputs to the mycumprodsum function.
    """
    my_delta = np.array(my_delta)

    # Determining maximum chunk size so that "delta_T / gamma^T" would be
    # guaranteed to be within the floating point range.
    aa = np.log(np.finfo(my_delta.dtype).max) - np.log(np.abs(my_delta).max())
    bb = -np.log(my_gamma)
    max_cumsum_len = int(aa / bb)

    if my_delta.size > max_cumsum_len:
        halflen = my_delta.size // 2
        delta_part1 = my_delta[:halflen]
        delta_part2 = my_delta[halflen:]
        out1 = mycumprodsum_numstable(delta_part1, my_gamma)
        out2 = mycumprodsum_numstable(delta_part2, my_gamma)
        out2 += (my_gamma ** np.arange(1, delta_part2.size + 1)) * out1[-1]
        output = np.concatenate([out1, out2])
    else:
        output = mycumprodsum(my_delta, my_gamma)

    return output

def add_vtarg_and_adv_np(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

    :param seg: (dict) the current segment of the trajectory (see traj_segment_generator return for more information)
    :param gamma: (float) Discount factor
    :param lam: (float) GAE factor
    """
    # last element is only used for last vtarg, but we already zeroed it if last new = 1
    episode_starts_all = seg["episode_starts"]
    vpred_all = seg["vpred"]
    rewards_all = seg["rewards"]
    seg_nextvpred = seg["nextvpred"]
    num_samples = len(rewards_all)

    # First, we should split each trajectory
    ep_start_idxs = np.where(episode_starts_all)[0].tolist()
    stwz = ((len(ep_start_idxs) > 0) and (ep_start_idxs[0] == 0))
    start_idxs = ep_start_idxs if stwz else ([0] + ep_start_idxs)
    end_idxs = start_idxs[1:] + [num_samples]

    adv_list = []
    for start_idx, end_idx in zip(start_idxs, end_idxs):
        rewards = rewards_all[start_idx:end_idx]
        vpred = vpred_all[start_idx:end_idx]
        vpred_next = np.empty_like(vpred)
        vpred_next[:-1] = vpred[1:]
        if end_idx == num_samples:
            vpred_next[-1] = seg_nextvpred
        else:
            vpred_next[-1] = 0.
        delta = rewards + gamma * vpred_next - vpred
        A_hat_t = mycumprodsum_numstable(delta[::-1], lam*gamma)[::-1]
        adv_list.append(A_hat_t)

    seg['adv'] = np.concatenate(adv_list, axis=0)
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def better_worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    seg_gen = None
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    # save final observation where user can get it, then reset
                    info['terminal_observation'] = observation
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == 'multiple_steps':
                (actor_weights_np, n_steps) = data
                if seg_gen is None:
                    seg_gen = traj_seg_roller(actor_weights_np, env, n_steps)
                output = seg_gen.__next__()
                remote.send(output)
            elif cmd == 'seed':
                remote.send(env.seed(data))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'render':
                remote.send(env.render(*data[0], **data[1]))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError
        except EOFError:
            break

class BetterSubprocVecEnv(SubprocVecEnv):
    def __init__(self, *args, **kwargs):
        super(BetterSubprocVecEnv, self).__init__(*args, **kwargs)

    def multiple_steps(self, actor_weights_np, n_steps):
        for remote in self.remotes:
            remote.send(('multiple_steps', (actor_weights_np, n_steps)))
        out_dicts = [remote.recv() for remote in self.remotes]
        return out_dicts

class BetterRunner(AbstractEnvRunner):
    def __init__(self, *, env, model, n_steps, gamma, lam, env_type='stepper'):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        env_type = 'roller' if isinstance(env, BetterSubprocVecEnv) else 'stepper'

        if env_type == 'stepper':
            super().__init__(env=env, model=model, n_steps=n_steps)
        elif env_type == 'roller':
            self.model = model
            self.env = env
            self.n_steps = n_steps
            msg_ = 'Make sure you set add_action_ph=True to the policy constructor'
            assert self.model.act_model.action_ph is not None, msg_

        self.lam = lam
        self.gamma = gamma
        self.env_type = env_type
        assert env_type in ('stepper', 'roller')

    def run(self):
        if self.env_type == 'stepper':
            return self.run_stepper()
        elif self.env_type == 'roller':
            return self.run_roller()
        else:
            raise RunTimeError(f'env_type {env_type} not implemented.')

    def run_stepper(self):
        """
        Run a learning step of the model

        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """
        # mb stands for minibatch
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        ep_infos = []
        for _ in range(self.n_steps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)
            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)
            mb_rewards.append(rewards)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        true_reward = np.copy(mb_rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
            delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values

        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = \
            map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward))

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward

    def run_roller(self):
        tf_session = self.model.sess
        policy = self.model.act_model
        actor_weights_np = get_tf_actor_weights(tf_session)
        traj_segments = self.env.multiple_steps(actor_weights_np, self.n_steps)
        ep_infos = []
        for seg in traj_segments:
            observations = seg['observations']
            episode_start = seg['episode_start']
            vpreds = self.model.value(observations)
            nextvpred = vpreds[-1] * (1- episode_start)
            seg['vpred'] = vpreds
            seg['nextvpred'] = nextvpred
            # add_vtarg_and_adv(seg, self.gamma, self.lam)
            # old_adv, old_tdlamret = seg.pop('adv'), seg.pop('tdlamret')
            add_vtarg_and_adv_np(seg, self.gamma, self.lam)
            # new_adv, new_tdlamret = seg['adv'], seg['tdlamret']
            # assert np.allclose(old_adv, new_adv), (old_adv, new_adv)
            # assert np.allclose(old_tdlamret, new_tdlamret), (old_tdlamret, new_tdlamret)
            for rrr, lll in zip(seg['ep_rets'], seg['ep_lens']):
                ep_infos.append({'l': lll, 'r':rrr})

            msg_ = f'{seg["rewards"].shape} != {(self.n_steps,)}'
            assert seg['rewards'].shape == (self.n_steps,), msg_

            msg_ = f'{seg["tdlamret"].shape} != {(self.n_steps,)}'
            assert seg['tdlamret'].shape == (self.n_steps,), msg_

            msg_ = f'{seg["observations"].shape[0]} != {self.n_steps}'
            assert seg['observations'].shape[0] == self.n_steps, msg_

            msg_ = f'{seg["dones"].shape} != {(self.n_steps,)}'
            assert seg['dones'].shape == (self.n_steps,), msg_

            msg_ = f'{seg["actions"].shape[0]} != {self.n_steps}'
            assert seg['actions'].shape[0] == self.n_steps, msg_

            msg_ = f'{seg["vpred"].shape} != {(self.n_steps,)}'
            assert seg['vpred'].shape == (self.n_steps,), msg_

        mb_obs = np.concatenate([seg['observations'] for seg in traj_segments], axis=0)
        mb_obs = mb_obs.reshape(mb_obs.shape[0], -1)

        mb_returns = np.concatenate([seg['tdlamret'] for seg in traj_segments], axis=0)

        mb_dones = np.concatenate([seg['dones'] for seg in traj_segments], axis=0)

        mb_actions = np.concatenate([seg['actions'] for seg in traj_segments], axis=0)
        mb_actions = mb_actions.reshape(mb_actions.shape[0], -1)

        mb_values = np.concatenate([seg['vpred'] for seg in traj_segments], axis=0)

        mb_neglogpacs = policy.sess.run(policy.neglogp_action_ph,
                                        {policy.obs_ph: mb_obs,
                                         policy.action_ph : mb_actions})

        mb_states = None

        true_reward = np.concatenate([seg['rewards'] for seg in traj_segments], axis=0)
        return (mb_obs, mb_returns, mb_dones, mb_actions, mb_values,
                mb_neglogpacs, mb_states, ep_infos, true_reward)

###############################################################################
################################# trpo_mpi/ppo1 ###############################
###############################################################################

def traj_segment_generator(policy, env, horizon, reward_giver=None, gail=False, env_type=None):
    if hasattr(env, 'multiple_steps'):
        env_type = 'roller'
    else:
        env_type = 'stepper'

    if env_type == 'stepper':
        return traj_segment_generator_stepper(policy, env, horizon,
                                              reward_giver=reward_giver, gail=gail)
    elif env_type == 'roller':
        return traj_segment_generator_roller(policy, env, horizon,
                                             reward_giver=reward_giver, gail=gail)
    else:
        raise ValueError(f'Unknow env_type {env_type}')

def traj_segment_generator_stepper(policy, env, horizon, reward_giver=None, gail=False):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

    :param policy: (MLPPolicy) the policy
    :param env: (Gym Environment) the environment
    :param horizon: (int) the number of timesteps to run per batch
    :param reward_giver: (TransitionClassifier) the reward predicter from obsevation and action
    :param gail: (bool) Whether we are using this generator for standard trpo or with gail
    :return: (dict) generator that returns a dict with the following keys:

        - observations: (np.ndarray) observations
        - rewards: (numpy float) rewards (if gail is used it is the predicted reward)
        - true_rewards: (numpy float) if gail is used it is the original reward
        - vpred: (numpy float) action logits
        - dones: (numpy bool) dones (is end of episode, used for logging)
        - episode_starts: (numpy bool)
            True if first timestep of an episode, used for GAE
        - actions: (np.ndarray) actions
        - nextvpred: (numpy float) next action logits
        - ep_rets: (float) cumulated current episode reward
        - ep_lens: (int) the length of the current episode
        - ep_true_rets: (float) the real environment reward
    """
    # Check when using GAIL
    assert not (gail and reward_giver is None), "You must pass a reward giver when using GAIL"

    # Initialize state variables
    step = 0
    action = env.action_space.sample()  # not used, just so we have the datatype
    observation = env.reset()

    cur_ep_ret = 0  # return in current episode
    current_it_len = 0  # len of current iteration
    current_ep_len = 0 # len of current episode
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # Episode lengths

    # Initialize history arrays
    observations = np.array([observation for _ in range(horizon)])
    true_rewards = np.zeros(horizon, 'float32')
    rewards = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    episode_starts = np.zeros(horizon, 'bool')
    dones = np.zeros(horizon, 'bool')
    actions = np.array([action for _ in range(horizon)])
    states = policy.initial_state
    episode_start = True  # marks if we're on first timestep of an episode
    done = False

    while True:
        action, vpred, states, _ = policy.step(observation.reshape(-1, *observation.shape), states, done)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if step > 0 and step % horizon == 0:
            yield {
                    "observations": observations,
                    "rewards": rewards,
                    "dones": dones,
                    "episode_starts": episode_starts,
                    "true_rewards": true_rewards,
                    "vpred": vpreds,
                    "actions": actions,
                    "nextvpred": vpred[0] * (1 - episode_start),
                    "ep_rets": ep_rets,
                    "ep_lens": ep_lens,
                    "ep_true_rets": ep_true_rets,
                    "total_timestep": current_it_len
            }
            _, vpred, _, _ = policy.step(observation.reshape(-1, *observation.shape))
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []
            ep_lens = []
            # Reset current iteration length
            current_it_len = 0
        i = step % horizon
        observations[i] = observation
        vpreds[i] = vpred[0]
        actions[i] = action[0]
        episode_starts[i] = episode_start

        clipped_action = action
        # Clip the actions to avoid out of bound error
        if isinstance(env.action_space, gym.spaces.Box):
            clipped_action = np.clip(action, env.action_space.low, env.action_space.high)

        if gail:
            reward = reward_giver.get_reward(observation, clipped_action[0])
            observation, true_reward, done, info = env.step(clipped_action[0])
        else:
            observation, reward, done, info = env.step(clipped_action[0])
            true_reward = reward
        rewards[i] = reward
        true_rewards[i] = true_reward
        dones[i] = done
        episode_start = done

        cur_ep_ret += reward
        cur_ep_true_ret += true_reward
        current_it_len += 1
        current_ep_len += 1
        if done:
            # Retrieve unnormalized reward if using Monitor wrapper
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                if not gail:
                    cur_ep_ret = maybe_ep_info['r']
                cur_ep_true_ret = maybe_ep_info['r']

            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(current_ep_len)
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            current_ep_len = 0
            if not isinstance(env, VecEnv):
                observation = env.reset()
        step += 1

def traj_segment_generator_roller(policy, env, horizon, reward_giver=None, gail=False):
    assert reward_giver is None, 'Not supported'
    assert gail is False, 'Not supported'

    episode_start = True
    ep_ret = 0
    ep_len = 0
    env.reset()
    while True:
        ep_rets = []
        ep_lens = []
        steps_taken = 0
        partial_outs = defaultdict(list)
        while steps_taken < horizon:
            n_steps = horizon - steps_taken
            d = env.multiple_steps(n_steps, policy, expl_noise=None, policy_lib='tf')
            for key, val in d.items():
                partial_outs[key].append(val)
            last_dones = d['dones']
            last_rewards = d['rewards']
            steps_taken = steps_taken + last_dones.size
            ep_ret = ep_ret + last_rewards.sum()
            ep_len = ep_len + last_dones.size
            if last_dones[-1]:
                ep_rets.append(ep_ret)
                ep_ret = 0
                ep_lens.append(ep_len)
                ep_len = 0
                env.reset()

        agg_dict = dict()

        ################# s/a/r/d ###################
        for key, val in partial_outs.items():
            agg_dict[key] = np.concatenate(val, axis=0)

        observations = agg_dict['observations']
        actions = agg_dict['actions']
        rewards = agg_dict['rewards']
        dones = agg_dict['dones']

        ################# vpreds ####################
        vpreds = policy.value(observations)

        ############### ep_starts ###################
        episode_starts = np.empty_like(dones)
        episode_starts[0] = episode_start
        episode_starts[1:] = dones[:-1]
        episode_start = dones[-1]

        ############### nextvpred ###################
        nextvpred = vpreds[-1] * (1- episode_start)

        ############# total_timestep ################
        total_timestep = dones.size

        yield {"observations": observations,
               "rewards": rewards,
               "dones": dones,
               "episode_starts": episode_starts,
               "true_rewards": rewards,
               "vpred": vpreds,
               "actions": actions,
               "nextvpred": nextvpred,
               "ep_rets": ep_rets,
               "ep_lens": ep_lens,
               "ep_true_rets": ep_rets,
               "total_timestep": total_timestep,
              }

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

    :param seg: (dict) the current segment of the trajectory (see traj_segment_generator return for more information)
    :param gamma: (float) Discount factor
    :param lam: (float) GAE factor
    """
    # last element is only used for last vtarg, but we already zeroed it if last new = 1
    episode_starts = np.append(seg["episode_starts"], False)
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    rew_len = len(seg["rewards"])
    seg["adv"] = np.empty(rew_len, 'float32')
    rewards = seg["rewards"]
    lastgaelam = 0
    for step in reversed(range(rew_len)):
        nonterminal = 1 - float(episode_starts[step + 1])
        delta = rewards[step] + gamma * vpred[step + 1] * nonterminal - vpred[step]
        seg["adv"][step] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


###############################################################################
###################################### td3 ####################################
###############################################################################

class UnlimitedTD3Policy(TD3FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        kwargs = dict(kwargs) # shallow copy
        self.do_mlp_output_tanh = kwargs.pop('do_mlp_output_tanh', True)
        self.mlp_output_scaling = kwargs.pop('mlp_output_scaling', 1)
        kwargs['layers'] = kwargs.get('layers', [64, 64])
        kwargs['act_fun'] = kwargs.get('act_fun', tf.nn.relu)
        super(UnlimitedTD3Policy, self).__init__(*args, **kwargs,
                                                 layer_norm=False,
                                                 feature_extraction="mlp")

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                pi_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                pi_h = tf.layers.flatten(obs)

            pi_h = mlpmaker(pi_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)

            activation = tf.nn.tanh if self.do_mlp_output_tanh else None

            env_actscale = (self.ac_space.high - self.ac_space.low) / 2.
            env_actscale_unq = np.unique(env_actscale)
            assert env_actscale_unq.size == 1, env_actscale
            output_scale = self.mlp_output_scaling / env_actscale_unq.item()
            self.policy = policy = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=activation) * output_scale

        return policy

# Patching stablebaselines
def patch_ppo2():
    import stable_baselines.common.vec_env.subproc_vec_env
    stable_baselines.common.vec_env.subproc_vec_env._worker = better_worker
    import stable_baselines.ppo2.ppo2
    stable_baselines.ppo2.ppo2.Runner = BetterRunner

def patch_trpo_ppo1():
    import stable_baselines.trpo_mpi.utils
    stable_baselines.trpo_mpi.utils.traj_segment_generator = traj_segment_generator
    stable_baselines.trpo_mpi.trpo_mpi.traj_segment_generator = traj_segment_generator
    stable_baselines.ppo1.pposgd_simple.traj_segment_generator = traj_segment_generator
