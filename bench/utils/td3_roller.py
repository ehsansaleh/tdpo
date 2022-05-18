import sys
import time
from collections import deque
import warnings

import numpy as np
import tensorflow as tf
###***
import copy
from collections import defaultdict
import random

from mpi4py import MPI
from stable_baselines.common.mpi_adam import MpiAdam
from typing import Optional
###***

from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.math_util import unscale_action, scale_action
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.sac.sac import get_vars
from stable_baselines.td3.policies import TD3Policy
from stable_baselines import logger


class TD3ME(OffPolicyRLModel):
    """
    Twin Delayed DDPG (TD3) Multi-Environment
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/pdf/1802.09477.pdf
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

    :param policy: (TD3Policy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values and Actor networks)
        it can be a function of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("polyak update" of the target networks, between 0 and 1)
    :param policy_delay: (int) Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param action_noise: (ActionNoise) the action noise type. Cf DDPG for the different action noise type.
    :param target_policy_noise: (float) Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: (float) Limit for absolute value of target policy smoothing noise.
    :param train_freq: (int) Update the model every `train_freq` steps.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param gradient_steps: (int) How many gradient update after each step
    :param random_exploration: (float) Probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for TD3 normally but can help exploring when using HER + TD3.
        This hack was present in the original OpenAI Baselines repo (DDPG + HER)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        Note: this has no effect on TD3 logging for now
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """
    def __init__(self, policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
                 learning_starts=100, train_freq=100, gradient_steps=100, batch_size=128,
                 tau=0.005, policy_delay=2, action_noise=None,
                 target_policy_noise=0.2, target_noise_clip=0.5,
                 random_exploration=0.0, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None,
                 critic_lr_scaling=1):

        ###***
        if isinstance(env, list):
            self.env_list = env
            env = env[0]
        else:
            self.env_list = [env]
            env = env

        if action_noise is None:
            action_noise = [None for env in self.env_list]
        if isinstance(action_noise, list):
            self.action_noise_list = action_noise
            action_noise = action_noise[0]
        else:
            self.action_noise_list = [action_noise]
            action_noise = action_noise
        self.num_envs = len(self.env_list)
        assert len(self.action_noise_list) == self.num_envs
        buffer_size = buffer_size * self.num_envs
        batch_size = batch_size * self.num_envs
        self.critic_lr_scaling = critic_lr_scaling
        ###***

        super(TD3ME, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose,
                                    policy_base=TD3Policy, requires_vec_env=False, policy_kwargs=policy_kwargs,
                                    seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        ###***
        from stable_baselines.common.base_class import _UnvecWrapper
        for env_idx, env in enumerate(self.env_list):
            if isinstance(env, VecEnv):
                if env.num_envs == 1:
                    self.env_list[env_idx] = _UnvecWrapper(env)
                else:
                    raise ValueError("Error: the model requires a non vectorized environment or a single vectorized"
                                     " environment.")
        ###***

        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.tau = tau
        self.gradient_steps = gradient_steps
        self.gamma = gamma
        self.action_noise = action_noise
        self.random_exploration = random_exploration
        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        self.graph = None
        self.replay_buffer = None
        self.episode_reward = None
        self.sess = None
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.params = None
        self.summary = None
        self.policy_tf = None
        self.full_tensorboard_log = full_tensorboard_log

        self.obs_target = None
        self.target_policy_tf = None
        self.actions_ph = None
        self.rewards_ph = None
        self.terminals_ph = None
        self.observations_ph = None
        self.action_target = None
        self.next_observations_ph = None
        self.step_ops = None
        self.target_ops = None
        self.infos_names = None
        self.target_params = None
        self.learning_rate_ph = None
        self.processed_obs_ph = None
        self.processed_next_obs_ph = None
        self.policy_out = None
        self.policy_train_op = None
        self.policy_loss = None

        if _init_setup_model:
            self.setup_model()

    ###***
    def set_random_seed(self, seed: Optional[int]) -> None:
        """
        :param seed: (Optional[int]) Seed for the pseudo-random generators. If None,
            do not change the seeds.
        """
        # Ignore if the seed is None
        if seed is None:
            return
        # Seed python, numpy and tf random generator
        from stable_baselines.common.misc_util import set_global_seeds
        set_global_seeds(seed)
        for env_idx, (env, action_noise) in enumerate(zip(self.env_list, self.action_noise_list)):
            if env is not None:
                env.seed(seed+1234*env_idx)
                # Seed the action space
                # useful when selecting random actions
                env.action_space.seed(seed+1234*env_idx)
            if action_noise is not None:
                action_noise.seed(seed+1234*env_idx)
        self.action_space.seed(seed)
    ###***

    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        # Rescale
        policy_out = unscale_action(self.action_space, self.policy_out)
        return policy.obs_ph, self.actions_ph, policy_out

    def setup_model(self):
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                if USE_ARRAY_RB:
                    action_dim = self.action_space.shape[0]
                    obs_dim = self.observation_space.shape[0]
                    self.replay_buffer = ArrayReplayBuffer(self.buffer_size, obs_dim, action_dim)
                else:
                    self.replay_buffer = ReplayBuffer(self.buffer_size)

                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                 **self.policy_kwargs)
                    self.target_policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                        **self.policy_kwargs)

                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    self.next_observations_ph = self.target_policy_tf.obs_ph
                    self.processed_next_obs_ph = self.target_policy_tf.processed_obs
                    self.action_target = self.target_policy_tf.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                     name='actions')
                    ###xxx
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    self.policy_out = policy_out = self.policy_tf.make_actor(self.processed_obs_ph)
                    # Use two Q-functions to improve performance by reducing overestimation bias
                    qf1, qf2 = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph)
                    # Q value when following the current policy
                    qf1_pi, _ = self.policy_tf.make_critics(self.processed_obs_ph,
                                                            policy_out, reuse=True)

                with tf.variable_scope("target", reuse=False):
                    # Create target networks
                    target_policy_out = self.target_policy_tf.make_actor(self.processed_next_obs_ph)
                    # Target policy smoothing, by adding clipped noise to target actions
                    target_noise = tf.random_normal(tf.shape(target_policy_out), stddev=self.target_policy_noise)
                    target_noise = tf.clip_by_value(target_noise, -self.target_noise_clip, self.target_noise_clip)
                    # Clip the noisy action to remain in the bounds [-1, 1] (output of a tanh)
                    noisy_target_action = tf.clip_by_value(target_policy_out + target_noise, -1, 1)
                    # Q values when following the target policy
                    qf1_target, qf2_target = self.target_policy_tf.make_critics(self.processed_next_obs_ph,
                                                                                noisy_target_action)

                with tf.variable_scope("loss", reuse=False):
                    # Take the min of the two target Q-Values (clipped Double-Q Learning)
                    min_qf_target = tf.minimum(qf1_target, qf2_target)

                    # Targets for Q value regression
                    q_backup = tf.stop_gradient(
                        self.rewards_ph +
                        (1 - self.terminals_ph) * self.gamma * min_qf_target
                    )

                    # Compute Q-Function loss
                    qf1_loss = tf.reduce_mean((q_backup - qf1) ** 2)
                    qf2_loss = tf.reduce_mean((q_backup - qf2) ** 2)

                    qvalues_losses = qf1_loss + qf2_loss

                    # Policy loss: maximise q value
                    self.policy_loss = policy_loss = -tf.reduce_mean(qf1_pi)

                    # Policy train op
                    # will be called only every n training steps,
                    # where n is the policy delay

                    ###xxx policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    ###xxx policy_train_op = policy_optimizer.minimize(policy_loss, var_list=get_vars('model/pi'))
                    ###xxx self.policy_train_op = policy_train_op
                    ###***
                    self.policy_grads = tf_util.flatgrad(self.policy_loss, tf_util.get_trainable_vars('model/pi/'))
                    self.policy_optimizer = MpiAdam(var_list=tf_util.get_trainable_vars('model/pi/'), beta1=0.9, beta2=0.999,
                                                    epsilon=1e-08)
                    ###***

                    # Q Values optimizer
                    ###xxx qvalues_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    ###xxx qvalues_params = get_vars('model/values_fn/')
                    self.critic_grads = tf_util.flatgrad(qvalues_losses, tf_util.get_trainable_vars('model/values_fn/'))
                    self.critic_optimizer = MpiAdam(var_list=tf_util.get_trainable_vars('model/values_fn/'), beta1=0.9, beta2=0.999,
                                                    epsilon=1e-08)

                    # Q Values and policy target params
                    source_params = get_vars("model/")
                    target_params = get_vars("target/")

                    # Polyak averaging for target variables
                    self.target_ops = [
                        tf.assign(target, (1 - self.tau) * target + self.tau * source)
                        for target, source in zip(target_params, source_params)
                    ]

                    # Initializing target to match source variables
                    target_init_op = [
                        tf.assign(target, source)
                        for target, source in zip(target_params, source_params)
                    ]

                    ###xxx train_values_op = qvalues_optimizer.minimize(qvalues_losses, var_list=qvalues_params)

                    self.infos_names = ['qf1_loss', 'qf2_loss']
                    # All ops to call during one training step
                    self.step_ops = [qf1_loss, qf2_loss,
                                     qf1, qf2
                                     ###xxx ,train_values_op
                                     ###***
                                     ,self.critic_grads
                                     ###***
                                     ]

                    # Monitor losses and entropy in tensorboard
                    tf.summary.scalar('policy_loss', policy_loss)
                    tf.summary.scalar('qf1_loss', qf1_loss)
                    tf.summary.scalar('qf2_loss', qf2_loss)
                    ###xxx tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

                # Retrieve parameters that must be saved
                self.params = get_vars("model")
                self.target_params = get_vars("target/")

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    ###***
                    self.policy_optimizer.sync()
                    self.critic_optimizer.sync()
                    ###***
                    self.sess.run(target_init_op)

                self.summary = tf.summary.merge_all()

    def _train_step(self, step, writer, learning_rate, update_policy):
        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch

        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(self.batch_size, -1),
            self.terminals_ph: batch_dones.reshape(self.batch_size, -1),
            ###xxx self.learning_rate_ph: learning_rate
        }

        step_ops = self.step_ops
        ###***
        ###***
        if update_policy:
            # Update policy and target networks
            step_ops = step_ops + [###xxx self.policy_train_op,
                                   ###xxx self.target_ops,
                                   ###***
                                   self.policy_grads,
                                   ###***
                                   self.policy_loss]

        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None:
            out = self.sess.run([self.summary] + step_ops, feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            out = self.sess.run(step_ops, feed_dict)

        # Unpack to monitor losses
        ###xxx qf1_loss, qf2_loss, *_values = out

        ###***
        if update_policy:
            qf1_loss, qf2_loss, qf1, qf2, critic_grads, policy_grads, policy_loss = out
        else:
            qf1_loss, qf2_loss, qf1, qf2, critic_grads = out
        with self.sess.as_default(), self.graph.as_default():
            self.critic_optimizer.update(critic_grads, learning_rate=learning_rate*self.critic_lr_scaling)
            if update_policy:
                # I know that the critic networks should be updated after the policy update
                # However, the following order made the mpi results
                # much closer to the original td3 script results.
                self.sess.run(self.target_ops)
                self.policy_optimizer.update(policy_grads, learning_rate=learning_rate)
        ###***

        return qf1_loss, qf2_loss

    def learn(self, total_timesteps, callback=None,
              log_interval=4, tb_log_name="TD3", reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn()

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            # Initial learning rate
            current_lr = self.learning_rate(1)

            start_time = time.time()

            ###***
            episode_rewards_list = []
            episode_lens_list = []
            last_logging_eps = 0
            #episode_successes_list = [[]  for env in self.env_list]
            #self.episode_reward = np.zeros((self.num_envs,)) #???
            #ep_info_buf_list = [deque(maxlen=100)  for env in self.env_list]


            leftovers_list = [None for env in self.env_list]
            ###***
            ###xxx episode_rewards = [0.0]
            ###xxx episode_successes = []
            ###xxx if self.action_noise is not None:
            ###xxx     self.action_noise.reset()
            ###xxx obs = self.env.reset()
            ###xxx self.episode_reward = np.zeros((1,))
            ###xxx ep_info_buf = deque(maxlen=100)
            n_updates = 0
            infos_values = []
            rank = MPI.COMM_WORLD.Get_rank()
            mpi_size = MPI.COMM_WORLD.Get_size()
            comm = MPI.COMM_WORLD

            myrank_timesteps = 0
            sofar_timesteps_allranks = 0

            roller_kwargs = dict(do_random_exploration=False,
                                 nsteps_random_exploration=0)
            seg_gen_list = [roller_seggen_td3(self.policy_tf, env, action_noise,
                                              self.train_freq, roller_kwargs)
                            for env, action_noise in
                            zip(self.env_list, self.action_noise_list)]

            while sofar_timesteps_allranks < total_timesteps:
                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy
                # if random_exploration is set to 0 (normal setting)
                ###***
                assert self.random_exploration == 0, 'not implemented'
                for pkg in zip(range(self.num_envs), self.env_list,
                               self.action_noise_list, seg_gen_list,
                               leftovers_list):

                    env_idx, env, action_noise, seg_gen, leftovers = pkg
                    do_rand_exp = self.num_timesteps < self.learning_starts
                    nsteps_random_exploration = self.learning_starts - self.num_timesteps
                    roller_kwargs['do_random_exploration'] = do_rand_exp
                    roller_kwargs['nsteps_random_exploration'] = nsteps_random_exploration
                    seg = seg_gen.__next__()

                    unscaled_actions = seg['actions'] # [low, high]
                    scaled_actions = scale_action(self.action_space, unscaled_actions) # [-1, 1]

                    observations = seg["observations"]
                    rewards = seg["rewards"]
                    dones = seg["dones"]
                    episode_starts = seg["episode_starts"]
                    ep_rets = seg["ep_rets"]
                    ep_lens = seg["ep_lens"]
                    ep_rets = seg["ep_true_rets"]
                    seg_timesteps = seg["total_timestep"]

                    # Store transition in the replay buffer.
                    if USE_ARRAY_RB:
                        if leftovers is not None:
                            last_obs, last_action, last_reward, last_done = leftovers
                            last_new_obs = observations[:1]
                            self.replay_buffer.add(last_obs, last_action, last_reward,
                                                    last_new_obs, last_done)

                        self.replay_buffer.add(observations[:-1], scaled_actions[:-1],
                                                rewards[:-1], observations[1:], dones[:-1])

                        leftovers = observations[-1:], scaled_actions[-1:], rewards[-1:], dones[-1:]
                        leftovers_list[env_idx] = leftovers
                    else:
                        if leftovers is not None:
                            last_obs, last_action, last_reward, last_done = leftovers
                            last_new_obs = observations[0]
                            self.replay_buffer.add(last_obs, last_action, last_reward,
                                                    last_new_obs, last_done)

                        for obs, action, reward, new_obs, done in zip(observations, scaled_actions,
                                                                      rewards, observations[1:],
                                                                      dones.astype(np.float64)):
                            self.replay_buffer.add(obs, action, reward, new_obs, done)

                        leftovers = observations[-1], scaled_actions[-1], rewards[-1], dones[-1]
                        leftovers_list[env_idx] = leftovers

                    episode_rewards_list = episode_rewards_list + ep_rets
                    episode_lens_list = episode_lens_list + ep_lens

                assert writer is None, 'Not implemented'
                ###***

                mb_infos_vals = []
                # Update policy, critics and target networks
                for grad_step in range(self.gradient_steps):
                    # Break if the warmup phase is not over
                    # or if there are not enough samples in the replay buffer
                    if not self.replay_buffer.can_sample(self.batch_size) \
                            or self.num_timesteps < self.learning_starts:
                        break
                    n_updates += 1
                    # Compute current learning_rate
                    frac = 1.0 - sofar_timesteps_allranks / total_timesteps
                    current_lr = self.learning_rate(frac)
                    # Update policy and critics (q functions)
                    # Note: the policy is updated less frequently than the Q functions
                    # this is controlled by the `policy_delay` parameter
                    mb_infos_vals.append(
                        self._train_step(sofar_timesteps_allranks, writer, current_lr,
                                         (myrank_timesteps + grad_step) % self.policy_delay == 0))

                # Log losses and entropy, useful for monitor training
                if len(mb_infos_vals) > 0:
                    infos_values = np.mean(mb_infos_vals, axis=0)

                ###***

                self.num_timesteps += seg_timesteps
                myrank_timesteps += seg_timesteps

                #########################################
                #episode_rewards_list, episode_lens_list
                num_episodes = len(episode_rewards_list)
                do_logging = False
                if rank == 0:
                    do_logging = (self.verbose >= 1)
                    do_logging = do_logging and (log_interval is not None)
                    if do_logging and ((num_episodes - last_logging_eps) >= log_interval):
                        last_logging_eps = num_episodes
                    else:
                        do_logging = False

                do_logging = MPI.COMM_WORLD.bcast(do_logging, root=0)
                sofar_timesteps_allranks = MPI.COMM_WORLD.allreduce(myrank_timesteps)
                sofar_episodes_allranks = MPI.COMM_WORLD.allreduce(num_episodes)
                if do_logging:
                    fps = int(sofar_timesteps_allranks / (time.time() - start_time))
                    combined_stats = dict()
                    if len(episode_rewards_list) > 0:
                        ep_rewmean = np.mean(episode_rewards_list[-1:])
                        mean_last_hundred_eprews = np.mean(episode_rewards_list[-100:])
                    else:
                        ep_rewmean = -np.inf
                        mean_last_hundred_eprews = -np.inf

                    if len(episode_lens_list) > 0:
                        eplenmean = np.mean(episode_lens_list)
                    else:
                        eplenmean = 0

                    combined_stats["100ep_rewmean"] = mean_last_hundred_eprews
                    combined_stats['ep_rewmean'] = ep_rewmean
                    combined_stats['eplenmean'] = eplenmean
                    combined_stats["n_updates"] = n_updates
                    combined_stats["current_lr"] = current_lr
                    combined_stats['time_elapsed'] = int(time.time() - start_time)
                    combined_stats["fps"] = fps

                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            combined_stats[name] = val

                    def as_scalar(scalar):
                        """
                        check and return the input if it is a scalar, otherwise raise ValueError

                        :param scalar: (Any) the object to check
                        :return: (Number) the scalar if x is a scalar
                        """
                        if isinstance(scalar, np.ndarray):
                            assert scalar.size == 1
                            return scalar[0]
                        elif np.isscalar(scalar):
                            return scalar
                        else:
                            raise ValueError('expected scalar, got %s' % scalar)

                    combined_stats_sums = MPI.COMM_WORLD.allreduce(
                        np.array([as_scalar(x) for x in combined_stats.values()]))
                    combined_stats = {k: v / mpi_size for (k, v) in
                                      zip(combined_stats.keys(), combined_stats_sums)}

                    # Total statistics.
                    combined_stats["total timesteps"] = sofar_timesteps_allranks
                    combined_stats["total episodes"] = sofar_episodes_allranks

                    for key_,val_ in combined_stats.items():
                        logger.logkv(key_, val_)

                    ###***
                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []



            return self

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        _ = np.array(observation)

        if actions is not None:
            raise ValueError("Error: TD3 does not have action probabilities.")

        # here there are no action probabilities, as DDPG does not use a probability distribution
        warnings.warn("Warning: action probability is meaningless for TD3. Returning None")
        return None

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions = self.policy_tf.step(observation)

        if self.action_noise is not None and not deterministic:
            actions = np.clip(actions + self.action_noise(), -1, 1)

        actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        actions = unscale_action(self.action_space, actions)  # scale the output for the prediction

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def get_parameter_list(self):
        return (self.params +
                self.target_params)

    def save(self, save_path, cloudpickle=False):
        data = {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "tau": self.tau,
            # Should we also store the replay buffer?
            # this may lead to high memory usage
            # with all transition inside
            # "replay_buffer": self.replay_buffer
            "policy_delay": self.policy_delay,
            "target_noise_clip": self.target_noise_clip,
            "target_policy_noise": self.target_policy_noise,
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "action_noise": self.action_noise,
            "random_exploration": self.random_exploration,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)

######################################################################
######################################################################
######################## Added for Multi-Step ########################
######################################################################
######################################################################

def unscale_action(action_space, action):
    """
    The multiple-action-rows generalization of
    stable_baselines.common.math_util.unscale_action.

    Rescale the action from [-1, 1] to [low, high]
    (no need for symmetric action space)

    :param action_space: (gym.spaces.box.Box)
    :param action: (np.ndarray)
    :return: (np.ndarray)
    """
    mid_np = (action_space.high + action_space.low) / 2.
    range_np = (action_space.high - action_space.low) / 2.
    if action.ndim == 2:
        mid_np = mid_np.reshape(1, -1)
        range_np = range_np.reshape(1, -1)
    action_scaled = action * range_np + mid_np
    return action_scaled

def scale_action(action_space, action):
    """
    The multiple-action-rows generalization of
    stable_baselines.common.math_util.scale_action.

    Rescale the action from [low, high] to [-1, 1]
    (no need for symmetric action space)

    :param action_space: (gym.spaces.box.Box)
    :param action: (np.ndarray)
    :return: (np.ndarray)
    """
    mid_np = (action_space.high + action_space.low) / 2.
    range_np = (action_space.high - action_space.low) / 2.
    if action.ndim == 2:
        mid_np = mid_np.reshape(1, -1)
        range_np = range_np.reshape(1, -1)
    action_unscaled = (action - mid_np) / range_np
    return action_unscaled

def mycumprodsum(my_delta, my_gamma):
    my_delta = np.array(my_delta)
    c = np.arange(my_delta.size)
    c = np.power(1./my_gamma, c)
    a = my_delta * c
    a = np.cumsum(a)
    a /= c
    return a

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
    bb = -np.log(np.abs(my_gamma))
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

class NoiseGenerator:
    def __init__(self, action_dim, rng_props=None):
        self.action_dim = action_dim
        self.np_random_list = None
        self.last_sample = None
        self.indecis = None
        self.rng_props = copy.deepcopy(rng_props) or {'type': 'pink', 'timestep': 0.01, 'rolloff_hz': 10}
        self.noise_data = None # TODO: add the measured noise data
        if self.noise_data is not None:
            self.noise_data = (self.noise_data - np.mean(noise_data, axis=0).reshape(1,2)) / np.std(noise_data, axis=0).reshape(1,2)
        self.rng_type = self.rng_props['type']
        if self.rng_type == 'ornstein':
            self.rng_type = 'OrnsteinUhlenbeck'
        if self.rng_type == 'normal':
            self.rng_type = 'white'
        self.steady_state_initialization = self.rng_props.get('steady_state_initialization', True)
        self.set_coeffs()

    def seed(self, seed=None):
        if seed is not None:
            seed_arr = seed +  np.arange(self.action_dim)
        else:
            seed_arr = [None] * self.action_dim
        assert len(seed_arr) == self.action_dim
        self.np_random_list = [np.random.RandomState(seed=seed) for seed in seed_arr]

    def reset(self):
        assert self.np_random_list is not None, f'you should seed first'
        self.last_sample = 0. # just to have a unified indicator of resetting.

        if self.steady_state_initialization:
            assert -1. < self.a_coeff < 1., f'there is no steady state if |a| >= 1'

        if self.rng_type == 'pink':
            steady_state_x_var = ((self.b_coeff**2) *(self.indep_rng_std**2))/(1-self.a_coeff**2)
            if self.steady_state_initialization:
                steady_state_x_std = np.sqrt(steady_state_x_var)
            else:
                steady_state_x_std = 0.
            self.last_sample = np.array([np_random.randn() * steady_state_x_std for _,np_random in enumerate(self.np_random_list)])
        elif self.rng_type == 'OrnsteinUhlenbeck':
            steady_state_x_var = ((self.b_coeff**2) * (self.indep_rng_std**2))/(1-self.a_coeff**2)
            if self.steady_state_initialization:
                steady_state_x_std = np.sqrt(steady_state_x_var)
            else:
                steady_state_x_std = 0.
            self.last_sample = np.array([np_random.randn() * steady_state_x_std for _,np_random in enumerate(self.np_random_list)])
        elif self.rng_type == 'measured':
            self.indecis = np.array([np_random.random_integers(noise_data.shape[0]) for np_random in self.np_random_list])

    def set_coeffs(self):
        if self.rng_type == 'white':
            #y_n = y_{n-1} * a + b * x_n
            #out = c * y_n
            self.a_coeff = 0.
            self.b_coeff = 1.
            self.c_coeff = 1.
            self.indep_rng_std = 1. # std(x_n)
        elif self.rng_type == 'uniform':
            #y_n = y_{n-1} * a + b * x_n
            #out = c * y_n
            self.a_coeff = 0.
            self.b_coeff = 1.
            self.c_coeff = 1.
            self.indep_rng_std = 1. # std(x_n)
        if self.rng_type == 'pink':
            #y_n = y_{n-1} * a + b * x_n
            #out = c * y_n
            k = self.rng_props['timestep'] * self.rng_props['rolloff_hz'] / (2 * np.pi)
            self.a_coeff = 1./(k+1.)
            self.b_coeff = k/(k+1.)
            self.c_coeff = np.sqrt(3. * ((k+1)**2 - 1)) / k
            self.indep_rng_std = np.sqrt(1./3.) # std(x_n)
        elif self.rng_type == 'OrnsteinUhlenbeck':
            assert not ('theta' in self.rng_props and 'rolloff_hz' in self.rng_props), 'only one of theta or rolloff_hz should be provided for OrnsteinUhlenbeck'

            self.dt = self.rng_props['timestep']
            self.sqrtdt = np.sqrt(self.dt)

            if 'theta' in self.rng_props:
                self.theta = self.rng_props['theta']
            elif 'rolloff_hz' in self.rng_props:
                assert self.rng_props['rolloff_hz'] * self.rng_props['timestep'] <= np.pi, f'max cutoff freq is pi/dt = {np.pi / self.rng_props["timestep"]}'
                # The max cutoff freq corresponds to a = 0, which makes y_n = b * x_n, which is white noise!
                tau = np.pi / self.rng_props['rolloff_hz']
                self.theta = 1. / tau

            if 'sigma' in self.rng_props:
                self.sigma = self.rng_props['sigma']
            else:
                #self.sigma = np.sqrt(2. / tau)
                self.sigma = np.sqrt(self.theta * (2. - self.theta * self.dt)) # solution to E_n[var(y_n)] = 1

            #y_n = y_{n-1} * a + b * x_n
            self.a_coeff = (1. - self.dt * self.theta)
            self.b_coeff = self.sigma * self.sqrtdt
            self.c_coeff = 1.
            self.indep_rng_std = 1. # std(x_n)
        if self.rng_type == 'measured':
            #y_n = y_{n-1} * a + b * x_n
            #out = c * y_n
            #WARNING: The following values are completely wrong!
            self.a_coeff = 0.
            self.b_coeff = 1.
            self.c_coeff = 1.
            self.indep_rng_std = 1. # std(x_n)

        c_eq_1 = np.ones(self.action_dim, dtype=np.float64)
        output_coeff = self.rng_props.get('output_coeff', 1.)
        output_coeff = np.array(output_coeff).reshape(-1) * c_eq_1
        output_coeff = output_coeff.reshape(1, self.action_dim)
        self.c_coeff *= output_coeff

    def get_std(self):
        if self.rng_type in ('measured',):
            return 1. # These are fixed to output an std of 1.
        my_var = (self.b_coeff**2) * (self.c_coeff**2).mean() * (self.indep_rng_std**2) / (1. - (self.a_coeff ** 2))
        return np.sqrt(my_var)


    def __call__(self, n_steps):
        assert self.last_sample is not None, f'your should reset me first!'
        assert n_steps > 0

        if self.rng_type == 'white':
            noise_ = np.concatenate([np_random.randn(n_steps, 1)
                                     for np_random in self.np_random_list], axis=1)
            out = noise_ * self.c_coeff
            assert out.shape == (n_steps, self.action_dim)
            return out
        elif self.rng_type == 'uniform':
            noise_ = np.concatenate([np_random.uniform(-np.sqrt(3.), np.sqrt(3.), size=(n_steps, 1))
                                     for np_random in self.np_random_list], axis=1)
            out = noise_ * self.c_coeff
            assert out.shape == (n_steps, self.action_dim)
            return out
        elif self.rng_type in ('OrnsteinUhlenbeck', 'pink'):
            if self.rng_type == 'OrnsteinUhlenbeck':
                # Main stochastic differential equation:
                #      y_{t+dt} - y_t = - dt * (y_t - mu) / tau + sigma * \sqrt{2/tau} * Normal(0,1)
                # Here we set the mu(i.e. mean) to be zero, and sigma(i.e. std) to be one.
                x = np.array([np_random.randn(n_steps) for _,np_random in enumerate(self.np_random_list)])
            elif self.rng_type == 'pink':
                x = np.array([np_random.uniform(-1., 1., size=(n_steps)) for _,np_random in enumerate(self.np_random_list)])
            else:
                raise ValueError(f'rng_type {self.rng_type} not implemented.')

            assert x.shape == (self.action_dim, n_steps)

            y_lst = []
            for i in range(self.action_dim):
                if self.a_coeff > 0.01:
                    # Solution 1) The solution to y_n = y_{n-1} * a + b * x_n
                    # is y_n = (a**n) * y_0 + \sum_{k=1}^{n} (a**(n−k)) * b_k
                    # https://mjo.osborne.economics.utoronto.ca/index.php/tutorial/index/1/fod/t
                    y_i = mycumprodsum_numstable(self.b_coeff * x[i], self.a_coeff)
                    y_i += (self.a_coeff ** np.arange(1, n_steps+1)) * self.last_sample[i]
                else:
                    # Solution 2) applying the basic iteration rule
                    y_i = np.zeros_like(x[i])
                    last_y = self.last_sample[i]
                    for t, x_n in enumerate(x[i]):
                        last_y = y_i[t] = self.a_coeff * last_y + self.b_coeff * x_n

                self.last_sample[i] = y_i[-1]
                y_lst.append(y_i.reshape(n_steps, 1))

            y = np.concatenate(y_lst, axis=1)
            out = y * self.c_coeff
            assert out.shape == (n_steps, self.action_dim)
            return out

        elif self.rng_type == 'measured':
            assert self.noise_data is not None
            out_lst = []
            while True:
                rmng_steps = n_steps - sum(x.shape[0] for x in out_lst)
                if rmng_steps <= 0:
                    break
                out_ = np.concatenate([self.noise_data[self.indecis:(self.indecis + rmng_steps), b%self.noise_data.shape[1]] for b in range(action_dim)], axis=1)
                self.indecis = (self.indecis + out_.shape[0]) % self.noise_data.shape[0]
                out_lst.append(out_)
            out = np.concatenate(out_lst, axis=0)
            out *= self.c_coeff
            assert out.shape == (n_steps, self.action_dim)
            return out
        else:
            raise Exception(f'Unknown RNG Type: {self.rng_type}')

def roller_seggen_td3(policy, env, action_noise, horizon, roller_policy_kwargs):
    assert isinstance(roller_policy_kwargs, dict)
    episode_start = True
    ep_ret = 0
    ep_len = 0
    env.reset()
    action_noise.reset()

    env.set_tf_policy(policy.sess, naming='td3')
    env.set_policy_calls -= 1 # Just not to mess up the rollout counts
    np_random = np.random.RandomState(seed=12345)
    mlp_input = np_random.randn(1000, env.obs_dim).astype(np.float64)
    python_output = policy.step(mlp_input)
    cpp_output = env.infer_mlp(mlp_input)
    cpp_output = scale_action(env.action_space, cpp_output)
    #python_output = unscale_action(env.action_space, python_output)
    assert np.allclose(cpp_output, python_output, atol=1e-06), \
           f'The C++/Python MLP difference is {np.abs(cpp_output - python_output).max()}.'

    obs_dim, act_dim, h1, h2 = env.obs_dim, env.act_dim, env.h1, env.h2
    null_policy = dict(fc1 = np.zeros((h1, obs_dim), dtype=np.float64),
                       fc2 = np.zeros((h2, h1), dtype=np.float64),
                       fc3 = np.zeros((act_dim, h2), dtype=np.float64),
                       fc1_bias = np.zeros((h1,), dtype=np.float64),
                       fc2_bias = np.zeros((h2,), dtype=np.float64),
                       fc3_bias = np.zeros((act_dim,), dtype=np.float64))
    env.set_np_policy(null_policy)
    env.set_policy_calls -= 1 # Just not to mess up the rollout counts
    cpp_output = env.infer_mlp(mlp_input)
    assert np.allclose(cpp_output, 0., atol=1e-06), 'The null policy is not generating zero actions.'

    while True:
        ep_rets = []
        ep_lens = []
        steps_taken = 0
        partial_outs = defaultdict(list)
        while steps_taken < horizon:
            n_steps = horizon - steps_taken
            do_random_exploration = roller_policy_kwargs['do_random_exploration']
            nsteps_random_exploration = roller_policy_kwargs['nsteps_random_exploration']
            if do_random_exploration:
                null_policy['policy_logstd'] = None
                np_random = action_noise.np_random_list[0]
                action_dim = env.action_space.shape[0]
                def unif_noisegen(n_steps_):
                    unif_shape = (n_steps_, action_dim)
                    unif_noise = np_random.uniform(-1, 1, unif_shape)
                    unif_noise_unscaled = unscale_action(env.action_space, unif_noise)
                    return unif_noise_unscaled
                d = env.multiple_steps(nsteps_random_exploration, null_policy,
                                       expl_noise=unif_noisegen, policy_lib='np')
            else:
                d = env.multiple_steps(n_steps, policy, expl_noise=action_noise,
                                       policy_lib='tf', naming='td3')
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
                action_noise.reset()

        agg_dict = dict()

        ################# s/a/r/d ###################
        for key, val in partial_outs.items():
            agg_dict[key] = np.concatenate(val, axis=0)

        observations = agg_dict['observations']
        actions = agg_dict['actions']
        rewards = agg_dict['rewards']
        dones = agg_dict['dones']

        ############### ep_starts ###################
        episode_starts = np.empty_like(dones)
        episode_starts[0] = episode_start
        episode_starts[1:] = dones[:-1]
        episode_start = dones[-1]

        ############# total_timestep ################
        total_timestep = dones.size

        yield {"observations": observations,
               "rewards": rewards,
               "dones": dones,
               "episode_starts": episode_starts,
               "true_rewards": rewards,
               "actions": actions,
               "ep_rets": ep_rets,
               "ep_lens": ep_lens,
               "ep_true_rets": ep_rets,
               "total_timestep": total_timestep,
              }


USE_ARRAY_RB = True

class ArrayReplayBuffer:
    def __init__(self, size, obs_dim, action_dim):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obses_t = np.empty((size, obs_dim), dtype=np.float64)
        self.actions = np.empty((size, action_dim), dtype=np.float64)
        self.rewards = np.empty((size,), dtype=np.float64)
        self.obses_tp1 = np.empty((size, obs_dim), dtype=np.float64)
        self.dones = np.empty((size,), dtype=np.float64)

        self._maxsize = size
        self._pushed_points = 0

    def __len__(self):
        return min(self._maxsize, self._pushed_points)

    @property
    def buffer_size(self):
        """float: Max capacity of the buffer"""
        return self._maxsize

    @property
    def next_idx(self):
        return self._pushed_points % self._maxsize

    def can_sample(self, n_samples):
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self):
        """
        Check whether the replay buffer is full or not.

        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (np.ndarray) the last observation
        :param action: (np.ndarray) the action
        :param reward: (np.ndarray) the reward of the transition
        :param obs_tp1: (np.ndarray) the current observation
        :param done: (np.ndarray) is the episode done
        """

        num_points = done.size
        pushed_sofar = 0

        assert obs_t.shape == (num_points, self.obs_dim)
        assert action.shape == (num_points, self.action_dim)
        assert obs_tp1.shape == (num_points, self.obs_dim)
        assert reward.shape == (num_points,)
        assert done.shape == (num_points,)

        while True:
            next_push_size = min(self._maxsize - self.next_idx, num_points - pushed_sofar)

            if next_push_size <= 0:
                break

            self.obses_t[self.next_idx: self.next_idx+next_push_size, :] = obs_t[pushed_sofar: pushed_sofar + next_push_size, :]
            self.actions[self.next_idx: self.next_idx+next_push_size, :] = action[pushed_sofar: pushed_sofar + next_push_size, :]
            self.obses_tp1[self.next_idx: self.next_idx+next_push_size, :] = obs_tp1[pushed_sofar: pushed_sofar + next_push_size, :]
            self.rewards[self.next_idx: self.next_idx+next_push_size] = reward[pushed_sofar: pushed_sofar + next_push_size]
            self.dones[self.next_idx: self.next_idx+next_push_size] = done[pushed_sofar: pushed_sofar + next_push_size]

            pushed_sofar += next_push_size
            self._pushed_points += next_push_size

    def _encode_sample(self, idxes):
        return self.obses_t[idxes], self.actions[idxes], self.rewards[idxes], self.obses_tp1[idxes], self.dones[idxes]

    def sample(self, batch_size, **_kwargs):
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        idxes = [random.randint(0, len(self) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
