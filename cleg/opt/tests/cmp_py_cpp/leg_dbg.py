import random, math, collections, copy, os, re
import numpy as np
import scipy.integrate, scipy.signal
import matplotlib.pyplot as plt
import matplotlib
import json
import subprocess
from dm_control import mujoco
from dm_control import suite
from dm_control import viewer
from dm_control import rl
from collections import OrderedDict
from collections import deque

import importlib
has_gym = importlib.util.find_spec('gym') is not None
if has_gym:
    from gym import spaces

import sys
from os.path import abspath, dirname
sys.path.append(dirname(abspath(__file__)))
import noise

###############################################################
################## C++ Validation Amendments ##################
###############################################################

DONT_USE_NP_SIGN = True
USE_EXACT_MOTOR_SOLVER = True
RCRD_AND_DELAY_SELF_JUMPSTATE = True
GET_S_BEFORE_JUMPSTATE_UPDATE = True
INCLUDE_OBS_NOISE_IN_OBS_BUFFER = True
DEEP_COPY_OBSERVATION = True

Debug_step_outer = False
Debug_step_inner = False
Debug_reward = False
Debug_simtask = False

"""
Why DONT_USE_NP_SIGN?
  1) np.sign can be zero for zero values.
  2) at the beginning, zero omega values are common.
  3) if (var == 0) can be unstable (and unnecessarily complicated) for C++ implementations,
     so we'll remove this behavior from python.

Why USE_EXACT_MOTOR_SOLVER?
  1) C++ uses exact solutions for Motor ODE.

Why RCRD_AND_DELAY_SELF_JUMPSTATE?
  1) The C++ implementation computes the rewards currently, and then delays them using a buffer.
  2) The Python environment delays the observations, and then computes current rewards for delayed
     observations.
  3) "jump_state" can be a function of jump_time and the non-delayed time.
  4) At the reward evaluation time, the C++ implementation does not have access to the future
     jump_state, unlike the python implementation (Think carefully about 1&2).
  5) Therefore, if we want to make python and C++ act the same, python needs to use a delayed and
     non-instant "jump_state" in reward computation.

Why INCLUDE_OBS_NOISE_IN_OBS_BUFFER?
  1) In the first few inner steps, due to observation delay, you get the same initial observation
     for a few steps.
  2) In each of the first few process_observation calls, a new noise is sampled in the python env.
     Therefore, the first few inner step calls report non-identical observations due to sampling of
     different noises.
  3) The C++ implementation adds the noise to mj_obs, and then pushes it to the observation delay buffer.
     Therefore, you tend to get identical first few observations even when do_obs_noise=True.
  4) Since solving this in C++ requires introducing too many histor variables and complicating the code,
     We are okay with having identical first few observations even when do_obs_noise=True.
  5) Therefore, if we want to make python and C++ act the same, we have to sample the noise and push it
     to the observatio delay buffer to make the first few first few observations identical.

Why DEEP_COPY_OBSERVATION?
  1) This is is actually a minor bug in the python environment!
  2) Due to observation delay and the way observation_buffer works, you get the same exact dictionary
     in the first few process_observation calls in the line:
         self.observation = self.observation_buffer.add(new_observation)
  3) We then add noise to the jump_state in the following line:
        self.observation['joint_state'] += np.hstack((theta_noise, omega_noise))
     Disclaimer: the actual line is a bit different, but does the same thing.
  4) Since we're changing self.observation['joint_state'] inplace, and self.observation_buffer returns
     the same dictionary in the first few add calls, the noise can get accumulated.
  5) This is not a serious problem since the initial omega is small and the noise is proportional to it.
  6) However, for complete python/C++ matching, have to resolve this in-place modification issue
     by replacing self.observation with a deep copy of itself right when we retrieve it from the
     observation_buffer.

Why GET_S_BEFORE_JUMPSTATE_UPDAT?
  1) In python, jump_state gets updated at the last minute in the step_outer function, and the updated
     version gets reported in the output observation array.
  2) Replicating the same behavior in C++ requires making new copies of the observation to be returned
     from the step function. This can be wasteful and practically unnecessary.
  4) Therefore, to make python and C++ match (and not have an off-by-one do_jump signal isseu), we have
     to use the old jump_state when returning the observation.

Why Debug_*?
  1) These flags just enable/disable printing information.
"""

gear_ratio = 23.3594
rpm_per_omega = gear_ratio * 60 / (2*np.pi)

def fcn_k50(tau_Nm):
    co = np.array([0.3518, -1.677, 186.0537]);
    tau_abs = np.abs(tau_Nm)
    tmp = co[0]*tau_abs**3 + co[1]*tau_abs**2 + co[2]*tau_abs
    if DONT_USE_NP_SIGN:
        tau_nm_sgn = (-1) if (tau_Nm < 0) else 1
        tau_units = tmp * tau_nm_sgn
    else:
        tau_units = tmp * np.sign(tau_Nm)
    return tau_units

def dyno_motor_model(tau_cmd_Nm, abs_omega):
    rpm = rpm_per_omega * abs_omega
    tau_cmd_Nm_abs = np.abs(tau_cmd_Nm)
    cmd_units = fcn_k50(tau_cmd_Nm_abs)

    max_cmd = 2000
    rpm_for_max_cmd = 4830
    cmd_per_rpm_slope = -0.368
    max_cmd_at_rpm = np.maximum(0, max_cmd + np.minimum(0, cmd_per_rpm_slope * (rpm - rpm_for_max_cmd)))

    cmd_units = np.minimum(cmd_units, max_cmd_at_rpm)

    X = np.array([1, cmd_units, rpm, cmd_units**2, cmd_units*rpm, rpm**2])
    model_coeff = np.array([-2.48090029e-01,  5.95753139e-03,  1.11332764e-04,
        -5.66257261e-07, -1.71874239e-07, -1.17675521e-08])

    tau_Nm = np.dot(model_coeff, X)
    if DONT_USE_NP_SIGN:
        tau_cmd_Nm_sign = (-1) if (tau_cmd_Nm < 0) else 1
        tau_Nm = tau_Nm * tau_cmd_Nm_sign
    else:
        tau_Nm = tau_Nm * np.sign(tau_cmd_Nm)
    return tau_Nm

# Motor saturation model taken from its datasheet
def naive_motor_model(tau_cmd_Nm, abs_omega):
    if abs_omega < 30.573932974138017:
        max_tau_avail = np.minimum(np.abs(tau_cmd_Nm),
                                   abs_omega*-1.37525388e-02 + 9.81094800e+00)
    elif abs_omega < 33.39821124007745:
        max_tau_avail = np.minimum(np.abs(tau_cmd_Nm),
                                   abs_omega*-2.48127817e-01 + 1.69767220e+01)
    elif abs_omega < 33.84650937752816:
        max_tau_avail = np.minimum(np.abs(tau_cmd_Nm),
                                   abs_omega*-1.93837450e+01 + 6.56072108e+02)
    else:
        max_tau_avail = 0
    if DONT_USE_NP_SIGN:
        if tau_cmd_Nm < 0:
            out_tau = -max_tau_avail
        else:
            out_tau = max_tau_avail
    else:
        out_tau = np.sign(tau_cmd_Nm)* max_tau_avail
    return out_tau

class HistoryBuffer():
    def __init__(self, taps=[0], min_length=1):
        """
        Create the buffer, specifying which time steps to draw history from:
        taps=[0 5 7], for example, produces a buffer with observations from the
        current time step, from 5 time steps ago, and from 7 time steps ago.

        The buffer is ordered left-to-right in time, so index 0 is the most
        recently added item (the current item) and higher indices are farther
        back in time.
        """
        self.taps = sorted(taps)
        self.max_length = max(self.taps[-1] + 1, min_length)
        self.buffer = deque()

    def add(self, item):
        """
        Add a new item to the buffer.

        Do this by appending on the left, and popping on the right if the
        buffer exceeds its maximum length.
        """
        self.buffer.appendleft(copy.deepcopy(item))
        if (len(self.buffer) > self.max_length):
            self.buffer.pop()

    def get(self, taps=None):
        """
        Get items from the buffer.

        By default, these items are at indices given by self.taps. If the
        optional argument taps is not None, then items are at indices given
        by taps.

        Taps at indices greater than or equal to the length of the buffer are
        copies of the last item in the buffer.
        """
        items = []
        for t in (taps if taps is not None else self.taps):
            if t < len(self.buffer):
                items.append(copy.deepcopy(self.buffer[t]))
            else:
                items.append(copy.deepcopy(self.buffer[-1]))
        return items

class DelayBuffer():
    def __init__(self, delay_steps=0):
        """
        delay_steps=0 produces a buffer with no delay. Higher numbers delay by that many steps.
        """
        self.max_length = delay_steps + 1
        self.buffer = []

    def add(self, item):
        """
        Add a new item to the buffer and return the oldest item.
        """
        self.buffer.append(copy.deepcopy(item))
        if len(self.buffer) > self.max_length:
            self.buffer = self.buffer[-self.max_length:]
        return self.buffer[0]

class FilterBuffer():
    def __init__(self, numtaps=0, window='triangle', cutoff=100, fs=None):
        """
        numtaps is the amount of history to use
        window is the shape of the filter window ('triangle', 'rectangle', or 'hamming')
        cutoff is the rolloff frequency in Hz for the Hamming window
        fs is the sampling frequency
        """
        self.numtaps = numtaps
        self.window = window
        self.cutoff = cutoff
        self.fs = fs
        self.buffer = []
        if self.window == 'hamming':
            self.coeff = scipy.signal.firwin(
                numtaps=self.numtaps,
                cutoff=self.cutoff,
                fs=self.fs,
            )
        elif self.window == 'rectangle':
            self.coeff = np.ones(self.numtaps, dtype=np.float64)
            self.coeff /= self.coeff.sum()
        elif self.window == 'triangle':
            self.coeff = np.arange(1, 1 + self.numtaps, dtype=np.float64)
            self.coeff /= self.coeff.sum()
        else:
            raise Exception(f'unknown window: {self.window}')

    def add(self, value):
        """
        Add a new value to the buffer and return the filtered value.
        value must be a numpy vector
        """
        self.buffer.append(value.copy())
        if len(self.buffer) > self.numtaps:
            self.buffer = self.buffer[-self.numtaps:]
        hist = np.array(self.buffer)
        missing_steps = len(self.coeff) - hist.shape[0]
        if missing_steps > 0:
            hist = np.vstack((np.tile(hist[0,:], (missing_steps, 1)), hist))
        estimate = np.dot(self.coeff, hist)
        return estimate

def solve_motor_ode(zeta, omega0, tau_cmd, y_init, ydot_init, t_f):
    """
    This function solves the homogeneous ode with constant coefficients

    `y_ddot + 2 * zeta * omega0 * y_dot + (omega0 ** 2) * (y - tau_cmd) = 0`
    """
    xdot_init = ydot_init
    x_init = y_init - tau_cmd

    assert np.abs(zeta) < 1.

    real_part = - zeta * omega0
    imag_part = np.sqrt(1-zeta*zeta) * omega0

    c1 = x_init
    c2 = (xdot_init - c1 * real_part) / imag_part

    e = np.exp(real_part * t_f)
    cos_, sin_ = np.cos(imag_part * t_f), np.sin(imag_part * t_f)
    x_t = e * (c1 * cos_ + c2 * sin_)
    xdot_t = e * (c1 * (real_part * cos_ - imag_part * sin_) + c2 * (real_part * sin_ + imag_part *cos_))

    y_t = x_t + tau_cmd
    ydot_t = xdot_t
    return y_t, ydot_t

class MotorModel():
    """
    Second-order motor model with approximately-correct overshoot and rise time.
    """

    def __init__(self, period=0.0014, overshoot=0.3):
        self.period = period
        self.overshoot = overshoot
        self.omega0 = 2*np.pi / period
        self.zeta = -np.log(self.overshoot) / np.sqrt(np.pi**2 + np.log(self.overshoot)**2)
        self.tau_state = np.zeros(4) # tau, tau_dot

    def command_torque(self, tau_cmd, delta_t):
        """
        Command the torque tau_cmd for delta_t seconds and return the applied torque.
        """
        tau = self.tau_state[0:2].copy() # return the current state, then integrate forward one timestep
        if USE_EXACT_MOTOR_SOLVER:
            self.tau_state[0], self.tau_state[2] = solve_motor_ode(self.zeta, self.omega0, tau_cmd[0],
                                                                   self.tau_state[0], self.tau_state[2], delta_t)
            self.tau_state[1], self.tau_state[3] = solve_motor_ode(self.zeta, self.omega0, tau_cmd[1],
                                                                   self.tau_state[1], self.tau_state[3], delta_t)
        else:
            soln = scipy.integrate.solve_ivp(lambda t, y: self._mc_f(t, y, tau_cmd), (0, delta_t), self.tau_state)
            self.tau_state = soln.y[:,-1]

        return tau

    def _mc_f(self, t, y, tau_cmd):
        x = y[0:2]
        x_dot = y[2:4]
        x_ddot = -2*self.zeta*self.omega0*x_dot + self.omega0**2*(tau_cmd - x)
        return np.hstack([x_dot, x_ddot])

class SimTask(suite.base.Task):
    def __init__(self, random=None):
        self.initial_state = None
        super(SimTask, self).__init__(random=random)

    def set_initial_state(self, initial_state):
        self.initial_state = initial_state

    def initialize_episode(self, physics):
        physics.named.data.qpos['slider'] = self.initial_state['pos_slider']
        physics.named.data.qvel['slider'] = self.initial_state['vel_slider']
        physics.named.data.qpos['hip'] = self.initial_state['theta_hip']
        physics.named.data.qpos['knee'] = self.initial_state['theta_knee']
        physics.named.data.qvel['hip'] = self.initial_state['omega_hip']
        physics.named.data.qvel['knee'] = self.initial_state['omega_knee']

    def _get_joint_state(self, physics):
        theta_hip = physics.named.data.qpos['hip'][0]
        theta_knee = physics.named.data.qpos['knee'][0]
        omega_hip = physics.named.data.qvel['hip'][0]
        omega_knee = physics.named.data.qvel['knee'][0]
        return (theta_hip, theta_knee, omega_hip, omega_knee)

    def _get_contact_state(self, physics):
        # foot-ground contact force
        foot_in_contact = False
        foot_force = np.zeros(3)
        for i_con, contact in enumerate(physics.data.contact):
            # each contact has format http://mujoco.org/book/APIreference.html#mjContact
            geom1 = contact[10]
            geom2 = contact[11]
            name1 = physics.model.id2name(geom1, 'geom')
            name2 = physics.model.id2name(geom2, 'geom')
            if name1 == 'ground' and name2 == 'lowerleg-limb':
                force_torque = np.zeros(6)
                mujoco.wrapper.mjbindings.mjlib.mj_contactForce(physics.model.ptr, physics.data.ptr, i_con, force_torque)
                force = force_torque[:3]
                contact_frame = np.array(contact[2]).reshape(3,3).T
                foot_force = contact_frame @ force
                foot_in_contact = True
        return foot_in_contact, foot_force

    def _get_cm_state(self, physics):
        # body masses
        body_masses = np.array([
            physics.named.model.body_mass['base'],
            physics.named.model.body_mass['upperleg'],
            physics.named.model.body_mass['lowerleg'],
        ])
        # center of mass positions
        body_poses = np.array([
            physics.named.data.xipos['base'],
            physics.named.data.xipos['upperleg'],
            physics.named.data.xipos['lowerleg'],
        ])
        # center of mass velocities
        body_vels = np.array([
            physics.named.data.cvel['base'][3:],
            physics.named.data.cvel['upperleg'][3:],
            physics.named.data.cvel['lowerleg'][3:],
        ])
        # whole-leg center of mass properties
        leg_mass = body_masses.sum()                         # leg total mass
        leg_pos = np.dot(body_masses, body_poses) / leg_mass # leg center of mass position
        leg_vel = np.dot(body_masses, body_vels) / leg_mass  # leg center of mass velocity
        return leg_mass, leg_pos, leg_vel

    def _get_site_velocity(self, physics, name):
        objtype = mujoco.wrapper.mjbindings.enums.mjtObj.mjOBJ_SITE
        objid = physics.model.name2id(name, 'site')
        res = np.zeros(6)
        flg_local = 0
        mujoco.wrapper.mjbindings.mjlib.mj_objectVelocity(physics.model.ptr, physics.data.ptr, objtype, objid, res, flg_local)
        return res[3:].copy()

    def get_observation(self, physics):
        foot_in_contact, foot_force = self._get_contact_state(physics)
        hip_position = physics.named.data.site_xpos['hip-center'].copy()
        knee_position = physics.named.data.site_xpos['knee-center'].copy()
        foot_position = physics.named.data.site_xpos['foot-center'].copy()
        hip_velocity = self._get_site_velocity(physics, 'hip-center')
        knee_velocity = self._get_site_velocity(physics, 'knee-center')
        foot_velocity = self._get_site_velocity(physics, 'foot-center')
        joint_state = np.array([*self._get_joint_state(physics)])
        tau_hip = physics.control()[0]
        tau_knee = physics.control()[1]
        try:
            # FIXME: copy() ?
            force_on_base = physics.named.data.sensordata['base-to-slider']
        except:
            force_on_base = np.zeros(3)
        leg_mass, leg_pos, leg_vel = self._get_cm_state(physics)
        return {
            'joint_state': np.array([*self._get_joint_state(physics)]),
            'joint_torque': np.array([tau_hip, tau_knee]),
            'foot_in_contact': float(foot_in_contact),
            'foot_force': foot_force,
            'hip_position': hip_position,
            'knee_position': knee_position,
            'foot_position': foot_position,
            'hip_velocity': hip_velocity,
            'knee_velocity': knee_velocity,
            'foot_velocity': foot_velocity,
            'foot_offset': foot_position - hip_position,
            'foot_offset_vel': foot_velocity - hip_velocity,
            'force_on_base': force_on_base,
            'leg_mass': leg_mass,
            'leg_pos': leg_pos,
            'leg_vel': leg_vel,
        }

    def get_reward(self, physics):
        return None

class SimEnv(rl.control.Environment):
    def __init__(self, sim_task, xml_string, control_timestep=0.00025):
        self.control_timestep = control_timestep
        physics = mujoco.Physics.from_xml_string(xml_string)
        if physics.timestep() > control_timestep:
            raise Exception(f'physics timestep {physics.timestep()} must be no greater than control timestep {control_timestep}')
        super(SimEnv, self).__init__(physics, sim_task, control_timestep=control_timestep)

    def reset(self, initial_state):
        self._task.set_initial_state(initial_state)
        return super(SimEnv, self).reset()

    def set_ground_contact(self, type='compliant'):
        solref = self._physics.named.model.pair_solref['ground-contact']
        solimp = self._physics.named.model.pair_solimp['ground-contact']
        with self._physics.reset_context():
            if type == 'compliant':
                solref[0] = 0.015
                solref[1] = 0.240
                solimp[0] = 0.010
                solimp[1] = 0.950
                solimp[2] = 0.001
                solimp[3] = 0.5
                solimp[4] = 9
            elif type == 'noncompliant':
                solref[0] = 0.002
                solref[1] = 1.000
                solimp[0] = 0.900
                solimp[1] = 0.950
                solimp[2] = 0.001
            else:
                raise Exception(f'unknown ground contact type {type}')

    def set_body_mass(self, name, mass):
        with self._physics.reset_context():
            self._physics.named.model.body_mass[name] = mass

    def set_body_inertia(self, name, inertia):
        with self._physics.reset_context():
            # important to copy into and not replace
            self._physics.named.model.body_inertia[name][:] = inertia

    def set_body_com(self, name, pos):
        with self._physics.reset_context():
            # important to copy into and not replace
            self._physics.named.model.body_ipos[name][:] = pos

    def set_joint_frictionloss(self, name, frictionloss):
        with self._physics.reset_context():
            self._physics.named.model.dof_frictionloss[name] = frictionloss

    def set_joint_damping(self, name, damping):
        with self._physics.reset_context():
            self._physics.named.model.dof_damping[name] = damping

    def set_joint_armature(self, name, armature):
        with self._physics.reset_context():
            self._physics.named.model.dof_armature[name] = armature

    def set_upperleg_length(self, length):
        with self._physics.reset_context():
            self._physics.named.model.geom_pos['upperleg-limb'][2] = -length/2
            self._physics.named.model.geom_size['upperleg-limb'][1] = length/2
            self._physics.named.model.body_pos['lowerleg'][2] = -length

    def set_lowerleg_length(self, length):
        with self._physics.reset_context():
            self._physics.named.model.geom_pos['lowerleg-limb'][2] = -length/2
            self._physics.named.model.geom_size['lowerleg-limb'][1] = length/2
            self._physics.named.model.site_pos['foot-center'][2] = -length

class SimInterface():
    def __init__(self, random=None, options={}, metadata={}):
        self.options = {
            # options - model
            'xml_file': 'leg.xml',
            'slurm_job_file': 'train_slurm_job.slurm',
            'ground_contact_type': 'compliant',

            # options - contraints
            'theta_hip_bounds': ((np.pi / 180) * np.array([-225, 90])).tolist(),
            'theta_knee_bounds': ((np.pi / 180) * np.array([-155, -35])).tolist(),
            'omega_hip_maxabs': 100,
            'omega_knee_maxabs': 100,
            'knee_minimum_z': 0.02,
            'hip_minimum_z': 0.08,
            'maxabs_tau': 10,
            'maxabs_omega_for_tau': 33.6, # for realism, should be 7500 rpm * 2 * pi / 60 / 23.3594 (gear ratio) = 33.6
            'tau_decay_constant': 5,
            'maxabs_desired_joint_angle': 10, # FIXME: this is arbitrary and is used only for OpenAI Gym compatibility

            # options - initial conditions
            'theta_hip_init': ((np.pi / 180) * np.array([-180, 30])).tolist(),
            'theta_knee_init': ((np.pi / 180) * np.array([-150, -40])).tolist(),
            'omega_hip_init': [-10, 10],
            'omega_knee_init': [-10, 10],
            'pos_slider_init': [0.4, 0.5],
            'vel_slider_init': [-0.1, 0.1],

            # options - reward
            'jump_vel_reward': False,
            'torque_smoothness_coeff': 500,
            'a_smooth_coeff': None,
            'a_osc_coeff': None,
            'constraint_penalty': 100,
            'turn_off_constraints': False,

            # options - task
            'contact_obs': False,
            'extra_obs_dim': 0,
            'action_is_delta': False,
            'stand_reward': 'dense',          # 'dense' or 'sparse'
            'posture_height': 0.20,
            'max_reach': 0.28,
            'time_before_reset': 2,           # maximum length of rollout, in seconds
            'obs_history_taps': [0],    # indices of past observations that will be returned by self.get_s
                                        # (e.g., [0] is current, [0, 5] is current and 5 time steps ago, etc.)
            'add_act_sm_r': False,    # adding action smoothness reward

            # options - task jump
            'do_jump': False,
            'jumping_obs': True,
            'jump_time_bounds': [1.1, 1.6],
            'timed_jump': False,
            'jump_push_time': 0.2,
            'jump_fly_time': 0.2,
            'max_leg_vel': 100,
            'jump_vel_coeff': 1000,

            # options - simulation
            'do_obs_noise': True,
            'omega_noise_scale': 1,
            'use_motor_model': True,
            'motor_saturation_model': 'dyno',
            'output_torque_scale': 0.8,
            'torque_delay': 0.001,            # torque delay (s)
            'observation_delay': 0.001,       # observation delay (s)

            # state filtering
            'filter_state': False,                  # whether to filter the state
            'filter_state_window': 'triangle',      # 'hamming', 'rectangle', 'triangle'
            'filter_state_numtaps': 80,             # number of history timesteps to use in the filter
            'filter_state_cutoff': 100,             # cutoff frequency for the filter (for hamming window)

            # action filtering
            'filter_action': False,                  # whether to filter the actions
            'filter_action_window': 'rectangle',     # 'hamming', 'rectangle', 'triangle'
            'filter_action_numtaps': 40,             # number of history timesteps to use in the filter
            'filter_action_cutoff': 100,             # cutoff frequency for the filter (for hamming window)

            # options - control
            'action_type': 'jointspace_pd',    # 'workspace_pd', 'jointspace_pd', or 'torque'
            'hip_kP': 1.25,
            'hip_kD': 0.05,
            'knee_kP': 1.25,
            'knee_kD': 0.05,
            'work_kP': 450,
            'work_kD': 7.5,
            'fd_velocities': False,
            'outer_loop_rate': 100,
            'inner_loop_rate': 4000,
        }
        deprecated_keys = [
            'obs_history_len',
            'osc_penalty',
        ]
        used_deprecated_keys = set(options.keys() & deprecated_keys)
        if len(used_deprecated_keys) > 0:
            print(f'WARNING: ignoring deprecated options: {used_deprecated_keys}')
        options = options.copy()
        for key in used_deprecated_keys:
            options.pop(key)
        extra_keys = options.keys() - self.options.keys()
        if len(extra_keys) > 0:
            raise Exception(f'Unknown options: {extra_keys}')
        self.nondefault_options = options.copy()
        self.options.update(options)

        # number of inner loops per outer loop
        if self.options['inner_loop_rate'] % self.options['inner_loop_rate'] != 0:
            raise Exception(f'inner loop rate ({self.options["inner_loop_rate"]}) must be an integer multiple of outer loop rate {self.options["outer_loop_rate"]}')
        self.inner_loops_per_outer_loop = self.options['inner_loop_rate'] // self.options['outer_loop_rate']

        # duration of inner loop step
        self.inner_step_time = 1 / self.options['inner_loop_rate']

        # observation history length
        self.obs_history_len = len(self.options['obs_history_taps'])

        # create simulation environment
        sim_task = SimTask(random=random)
        self.xml_string = open(self.options['xml_file']).read()
        if self.options['slurm_job_file'] is not None:
            self.slurm_job_file = open(self.options['slurm_job_file']).read()
        else:
            self.slurm_job_file = ''
        self.sim_env = SimEnv(sim_task, xml_string=self.xml_string, control_timestep=self.inner_step_time)

        # alter physics model
        self.sim_env.set_ground_contact(self.options['ground_contact_type'])

        # try:
        #     self.commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()
        # except:
        #     self.commit_hash = 'no git repo found'
        self.commit_hash = 'no git repo found'
        self.env_file_string = open(__file__).read()
        self.metadata = {
            'xml_string': self.xml_string,
            'slurm_job_file': self.slurm_job_file,
            'commit_hash': self.commit_hash,
            'env_file': self.env_file_string,
        }
        self.metadata.update(metadata)

    def get_options(self):
        return self.options

    def get_nondefault_options(self):
        return self.nondefault_options

    def get_metadata(self):
        return self.metadata

    def reset(self, initial_state=None, np_random=np.random, disable_randomizable=False):
        obs_dim, act_dim = self.get_dims()
        self.inner_step_count = 0
        self.outer_step_count = 0
        if self.options['filter_state']:
            self.joint_state_filter = FilterBuffer(
                numtaps=self.options['filter_state_numtaps'],
                window=self.options['filter_state_window'],
                cutoff=self.options['filter_state_cutoff'],
                fs=self.options['inner_loop_rate'],
            )
        if self.options['filter_action']:
            self.action_filter = FilterBuffer(
                numtaps=self.options['filter_action_numtaps'],
                window=self.options['filter_action_window'],
                cutoff=self.options['filter_action_cutoff'],
                fs=self.options['inner_loop_rate'],
            )
        self.motor = MotorModel()
        self.joint_torque_buffer = DelayBuffer(delay_steps=round(self.options['torque_delay'] / self.inner_step_time))
        self.observation_buffer = DelayBuffer(delay_steps=round(self.options['observation_delay'] / self.inner_step_time))
        if RCRD_AND_DELAY_SELF_JUMPSTATE:
            self.jump_state_buffer = DelayBuffer(delay_steps=round(self.options['observation_delay'] / self.inner_step_time)-1)
        self.obs_history_buffer = HistoryBuffer(
            taps=self.options['obs_history_taps'],
            min_length=2,   # so we have enough history to compute torque smoothness reward
        )
        if self.options['do_obs_noise']:
            self.omega_noise_generator = noise.MeasuredNoiseGenerator()
            #self.omega_noise_generator = noise.PinkNoiseGenerator(2, timestep=0.00025, rolloff_hz=100)
        self.jump_state = 0
        if disable_randomizable:
            self.jump_time = self.options['jump_time_bounds'][0]    # use minimum value
        else:
            self.jump_time = np_random.uniform(*self.options['jump_time_bounds'])
        self.leg_val_peak = 0
        self.has_touched_ground = False
        self.joint_torque_command = np.zeros(2)
        self.torque_components = np.zeros(4)
        self.joint_state_old = None
        self.leg_vel_peak = 0
        self.a = np.zeros(act_dim)
        self.a_raw = np.zeros(act_dim)
        self.a_old = None
        self.a_old2 = None
        self.reward = {}
        self.r_inner = 0.0
        self.simulation_error = False       # this flag is set to True if the simulation engine fails

        if initial_state is None:
            if disable_randomizable:
                initial_state = {
                    'theta_hip': (self.options['theta_hip_init'][0] +
                        self.options['theta_hip_init'][1])/2,
                    'theta_knee': (self.options['theta_knee_init'][0] +
                        self.options['theta_knee_init'][1])/2,
                    'omega_hip': (self.options['omega_hip_init'][0] +
                        self.options['omega_hip_init'][1])/2,
                    'omega_knee': (self.options['omega_knee_init'][0] +
                        self.options['omega_knee_init'][1])/2,
                    'pos_slider': (self.options['pos_slider_init'][0] +
                        self.options['pos_slider_init'][1])/2,
                    'vel_slider': (self.options['vel_slider_init'][0] +
                        self.options['vel_slider_init'][1])/2,
                }
            else:
                initial_state = {
                    'theta_hip': np_random.uniform(*self.options['theta_hip_init']),
                    'theta_knee': np_random.uniform(*self.options['theta_knee_init']),
                    'omega_hip': np_random.uniform(*self.options['omega_hip_init']),
                    'omega_knee': np_random.uniform(*self.options['omega_knee_init']),
                    'pos_slider': np_random.uniform(*self.options['pos_slider_init']),
                    'vel_slider': np_random.uniform(*self.options['vel_slider_init']),
                }
        time_step = self.sim_env.reset(initial_state)
        self.process_observation(time_step.observation)
        self.desired_foot_offset = self.observation['foot_offset'].copy()
        self.desired_joint_angle = self.observation['joint_state'][0:2].copy()
        self.data_buffer = [self.get_data()]

        return self.get_s()

    def get_data(self):
        return {
            'inner_step_count': self.inner_step_count,
            'outer_step_count': self.outer_step_count,
            'observation': self.observation,
            'joint_state_raw': self.observation['joint_state_raw'],
            'joint_state': self.observation['joint_state'],
            'jump_state': self.jump_state,
            'a': self.a,
            'a_raw': self.a_raw,
            'desired_foot_offset': self.desired_foot_offset,
            'desired_joint_angle': self.desired_joint_angle,
            'joint_torque_command': self.joint_torque_command,
            'torque_components': self.torque_components,
            'reward': self.reward,
            'r_inner': self.r_inner,
            'is_jumping': 1.0 if self.jump_state == 1 else 0.0,
        }

    def get_timestep(self):
        return self.inner_loops_per_outer_loop * self.inner_step_time

    def get_time(self):
        return self.inner_step_time * self.inner_step_count

    def time_greater_equal(self, t):
        tcur = self.get_time()
        if tcur > t:
            return True
        elif np.isclose(tcur, t, rtol=1e-4, atol=(1e-4 * self.inner_step_time)):
            return True
        else:
            return False

    def process_observation(self, new_observation):
        self.real_observation = new_observation.copy()
        if INCLUDE_OBS_NOISE_IN_OBS_BUFFER and self.options['do_obs_noise']:
            new_observation['obs_noise'] = copy.deepcopy(self.omega_noise_generator.sample())
        self.observation = self.observation_buffer.add(new_observation)
        if DEEP_COPY_OBSERVATION:
            self.observation = copy.deepcopy(self.observation)
        if RCRD_AND_DELAY_SELF_JUMPSTATE:
            old_jump_state = self.jump_state_buffer.add(copy.deepcopy(self.jump_state))
            self.observation['jump_state'] = old_jump_state
        if self.options['do_obs_noise']:
            joint_state_pre_noise = self.observation['joint_state']
            self.observation['joint_state_pre_noise'] = joint_state_pre_noise
            theta = joint_state_pre_noise[0:2]
            omega = joint_state_pre_noise[2:4]
            omega_noise_scale = np.minimum(1, np.abs(omega)) * self.options['omega_noise_scale']
            theta_noise = np.zeros(2)
            if INCLUDE_OBS_NOISE_IN_OBS_BUFFER:
                omega_noise = omega_noise_scale * self.observation['obs_noise']
            else:
                omega_noise = omega_noise_scale * self.omega_noise_generator.sample()
            self.observation['joint_state'] = joint_state_pre_noise + np.hstack((theta_noise, omega_noise))

        self.observation['joint_state_raw'] = self.observation['joint_state'].copy()
        if self.options['filter_state']:
            self.observation['joint_state'] = self.joint_state_filter.add(self.observation['joint_state'])

        self.observation['foot_offset_true'] = self.observation['foot_offset'].copy()
        self.observation['foot_offset_vel_true'] = self.observation['foot_offset_vel'].copy()

        # FIXME: we should use the joint_state from new_observation
        # here so that we don't get a double-delay on foot_offset. But
        # at the moment we don't apply noise directly on
        # new_observation, so we'd need to change that first.
        q, dq = joint_state_to_angles(self.observation['joint_state'])
        p, v = foot_p_and_v(q, dq)
        self.observation['foot_offset'] = p
        self.observation['foot_offset_vel'] = v

        q_raw, dq_raw = joint_state_to_angles(self.observation['joint_state_raw'])
        p_raw, v_raw = foot_p_and_v(q_raw, dq_raw)
        self.observation['foot_offset_raw'] = p_raw
        self.observation['foot_offset_vel_raw'] = v_raw

        if self.observation['foot_in_contact'] and not self.has_touched_ground:
            self.has_touched_ground = True

        # FIXME: there are at least three problems here:
        #
        #   1) For some jump options, the jump state will update based on
        #      quantities that are not observable to the agent (e.g., hip
        #      position). For training, this is ok - for testing, not so.
        #
        #   2) If we decide it is ok for the jump state to update based on
        #      quantities that are not observable to the agent, then we should
        #      be using new_observation (non-delayed) and not observation.
        #
        #   3) Think carefully about whether this state machine should be
        #      running at the inner loop rate - which it is now - or at the
        #      outer loop rate. In experiment, it runs at the inner loop rate,
        #      but the agent (of course) responds at the outer loop rate, and
        #      furthermore the jump signal is actually recorded at the outer
        #      loop rate - so...
        #
        #   ** for now, I moved the 'timed_jump' state machine to step_outer. **
        #
        if self.options['do_jump'] and not self.options['timed_jump']:
            #
            # FIXME: timing won't match experiment
            #
            if self.jump_state == 0:    # drop and stand
                if self.time_greater_equal(self.jump_time):
                    self.jump_state = 1
            elif self.jump_state == 1:  # jump
                if self.observation['hip_position'][2] > self.options['max_reach']:
                    self.jump_state = 2
            elif self.jump_state == 2:  # liftoff
                self.jump_state = 3
            elif self.jump_state == 3:  # in air, far from ground
                if self.observation['hip_position'][2] < self.options['max_reach']:
                    self.jump_state = 4
                if self.options['jump_vel_reward']:
                    self.jump_state = 5
            elif self.jump_state == 4:  # in air, close to ground
                pass
            elif self.jump_state == 5:  # in air, for timed period
                if self.time_greater_equal(self.jump_time + self.options['jump_push_time'] + self.options['jump_fly_time']):
                    self.jump_state = 6
            elif self.jump_state == 6:  # terminate
                pass
            else:
                raise Exception(f'invalid jump state {self.jump_state}')

        self.obs_history_buffer.add(self.observation)

    def step(self, a):
        return self.step_outer(a)

    def step_outer(self, a):
        if Debug_step_outer:
            print(f'Step outer {self.outer_step_count}:')
        self.a_raw = a

        # inner loop
        r = 0.0
        self.data_buffer = []
        for step in range(self.inner_loops_per_outer_loop):
            self.step_inner()
            r += self.r_inner / self.inner_loops_per_outer_loop         # average reward over all inner loop steps
            self.data_buffer.append(self.get_data())
        self.outer_step_count += 1

        # update history
        self.joint_state_old = self.observation['joint_state'].copy()
        self.a_old2 = self.a_old
        self.a_old = self.a.copy()

        # update jump state (moved from inner loop - see FIXME in process_observation)
        if GET_S_BEFORE_JUMPSTATE_UPDATE:
            s = self.get_s()
        if self.options['do_jump'] and self.options['timed_jump']:
            if self.jump_state == 0:    # drop and stand
                if self.time_greater_equal(self.jump_time):
                    self.jump_state = 1
            elif self.jump_state == 1:
                if self.time_greater_equal(self.jump_time + self.options['jump_push_time']):
                    self.jump_state = 5
            elif self.jump_state == 5:  # in air, for timed period
                if self.time_greater_equal(self.jump_time + self.options['jump_push_time'] + self.options['jump_fly_time']):
                    self.jump_state = 6
            elif self.jump_state == 6:  # terminate
                pass
            else:
                raise Exception(f'invalid jump state {self.jump_state}')

        # return state, reward, and flag if task is done
        if not GET_S_BEFORE_JUMPSTATE_UPDATE:
            s = self.get_s()
        done = self.time_greater_equal(self.options['time_before_reset'])
        if self.jump_state >= 6:
            done = True

        if Debug_step_outer:
            print(f'Reward: %+05.8f' % r)
            print(f'Done: {done}!')
            print('-'*20)

        return s, r, done

    def step_inner(self):
        if Debug_step_inner:
            print(f'Step inner {self.inner_step_count}:')
        if self.options['filter_action']:
            self.a = self.action_filter.add(self.a_raw)
        else:
            self.a = self.a_raw

        if Debug_step_inner:
            print(f'  --> 0) theta_dlyd           = %+05.8f, %+05.8f' % tuple(self.observation["joint_state_raw"][0:2]))
            print(f'  -->    omega_dlyd           = %+05.8f, %+05.8f' % tuple(self.observation["joint_state_raw"][2:4]))

        if self.options['action_type'] == 'torque':
            tau = self.a.copy()
        elif self.options['action_type'] == 'jointspace_pd':
            if self.options['action_is_delta']:
                self.desired_joint_angle = self.observation['joint_state_raw'][:2] + self.a
            else:
                self.desired_joint_angle = self.a.copy()
            tau = self.jointspace_pd(self.desired_joint_angle, self.observation['joint_state_raw'][0:2], self.observation['joint_state_raw'][2:4])
        elif self.options['action_type'] == 'workspace_pd':
            if self.options['action_is_delta']:
                self.desired_foot_offset = self.observation['foot_offset'] + np.array([self.a[0], 0, self.a[1]])
            else:
                self.desired_foot_offset = np.array([self.a[0], 0, self.a[1]])
            tau = self.workspace_pd(self.desired_foot_offset, self.observation['joint_state_raw'][0:2], self.observation['foot_offset_raw'], self.observation['foot_offset_vel_raw'])
        else:
            raise Exception("unknown action_type: " + self.options['action_type'])
        if Debug_step_inner:
            print(f'  --> 1) tau                  = %+05.8f, %+05.8f' % tuple(tau))
        self.joint_torque_command = tau

        # actuator delay
        self.joint_torque_current = self.joint_torque_buffer.add(self.joint_torque_command)
        if Debug_step_inner:
            print(f'  --> 2) joint_torque_current = %+05.8f, %+05.8f' % tuple(self.joint_torque_current)) #'= {self.joint_torque_current}')
        if self.options['use_motor_model']:
            self.joint_torque = self.motor.command_torque(self.joint_torque_current, self.inner_step_time)
        else:
            self.joint_torque = self.joint_torque_current.copy()

        if Debug_step_inner:
            print(f'  --> 3) joint_torque         = %+05.8f, %+05.8f' % tuple(self.joint_torque))

        self.joint_torque = self.joint_torque.copy()
        abs_omega_hip = np.abs(self.real_observation['joint_state'][2])
        abs_omega_knee = np.abs(self.real_observation['joint_state'][3])

        if self.options['motor_saturation_model'] == 'dyno':
            self.joint_torque[0] = dyno_motor_model(self.joint_torque[0], abs_omega_hip)
            self.joint_torque[1] = dyno_motor_model(self.joint_torque[1], abs_omega_knee)
        elif self.options['motor_saturation_model'] == 'naive':
            self.joint_torque[0] = naive_motor_model(self.joint_torque[0], abs_omega_hip)
            self.joint_torque[1] = naive_motor_model(self.joint_torque[1], abs_omega_knee)
        elif self.options['motor_saturation_model'] == 'legacy':
            maxabs_omega_for_tau = 21.66 #30.79
            tau_omega_slope = -0.397
            maxabs_tau_hip = np.maximum(0, self.options['maxabs_tau'] + np.minimum(0, tau_omega_slope * (abs_omega_hip - maxabs_omega_for_tau)))
            maxabs_tau_knee = np.maximum(0, self.options['maxabs_tau'] + np.minimum(0, tau_omega_slope * (abs_omega_knee - maxabs_omega_for_tau)))
            self.joint_torque[0] = np.clip(self.joint_torque[0], -maxabs_tau_hip, maxabs_tau_hip)
            self.joint_torque[1] = np.clip(self.joint_torque[1], -maxabs_tau_knee, maxabs_tau_knee)
        else:
            raise Exception(f'Unknown motor saturation model: {self.options["motor_saturation_model"]}')

        if Debug_step_inner:
            print(f'  --> 4) joint_torque_capped  = %+05.8f, %+05.8f' % tuple(self.options["output_torque_scale"] * self.joint_torque))

        if self.simulation_error:
            # the simulation has failed
            self.process_observation(copy.deepcopy(self.observation))
        else:
            # advance simulation one time step
            try:
                time_step = self.sim_env.step(self.options['output_torque_scale'] * self.joint_torque)
                self.process_observation(time_step.observation)
            except rl.control.PhysicsError:
                self.simulation_error = True
                self.process_observation(copy.deepcopy(self.observation))

        self.update_reward()

        self.inner_step_count += 1

    def jointspace_pd(self, q_des, q, dq):
        # FIXME: may need to scale torques (e.g., by 0.8) to match hardware
        tau_hip_p  = - self.options['hip_kP']  * (q[0] - q_des[0])
        tau_knee_p = - self.options['knee_kP'] * (q[1] - q_des[1])
        tau_hip_d  = - self.options['hip_kD']  * dq[0]
        tau_knee_d = - self.options['knee_kD'] * dq[1]
        self.torque_components = np.array([tau_hip_p, tau_knee_p, tau_hip_d, tau_knee_d])
        tau_hip = tau_hip_p + tau_hip_d
        tau_knee = tau_knee_p + tau_knee_d
        return np.array([tau_hip, tau_knee])

    def workspace_pd(self, x_des, q, x, v):
        q3 = np.array([0, q[0], q[1]])
        J = jacobian(q3)
        force_p = - self.options['work_kP'] * (x - x_des)
        force_d = - self.options['work_kD'] * v
        tau_p3 = J.T @ force_p
        tau_d3 = J.T @ force_d
        tau_p2 = np.array([tau_p3[1], tau_p3[2]])
        tau_d2 = np.array([tau_d3[1], tau_d3[2]])
        self.torque_components = np.array([*tau_p2, *tau_d2])
        tau = tau_p2 + tau_d2
        return tau

    def get_s(self):
        vars = []
        for observation_ in self.obs_history_buffer.get():
            vars.append(observation_['joint_state'])
        if self.options['contact_obs']:
            vars.append(self.observation['foot_in_contact'])
        if self.options['jumping_obs']:
            vars.append(1.0 if self.jump_state == 1 else 0.0)
        if self.options['extra_obs_dim'] > 0:
            vars.append(np.zeros(self.options['extra_obs_dim']))
        s = np.hstack(vars)
        return s.copy()

    def get_dims(self):
        obs_dim = 4 * self.obs_history_len
        if self.options['contact_obs']:
            obs_dim += 1
        if self.options['jumping_obs']:
            obs_dim += 1
        obs_dim += self.options['extra_obs_dim']
        act_dim = 2
        return (obs_dim, act_dim)

    def get_specs(self):
        """
        It's not at all clear what the bounds should be on the action. It is the
        desired joint angle. Should we restrict desired joint angle to the same
        bounds as joint angle? No, of course not, because the joint angle error
        will very rarely be zero. Smart controllers may choose desired joint
        angles that are well outside joint angle bounds in order to generate
        particular torques. As a placeholder, we will arbitrarily define large
        bounds on the action.

        FIXME
        """

        low = [
            self.options['theta_hip_bounds'][0],
            self.options['theta_knee_bounds'][0],
            -self.options['omega_hip_maxabs'],
            -self.options['omega_knee_maxabs'],
        ]
        high = [
            self.options['theta_hip_bounds'][1],
            self.options['theta_knee_bounds'][1],
            self.options['omega_hip_maxabs'],
            self.options['omega_knee_maxabs'],
        ]
        low = low * self.obs_history_len
        high = high * self.obs_history_len
        if self.options['contact_obs']:
            low.append(0)
            high.append(1)
        if self.options['jumping_obs']:
            low.append(0)
            high.append(1)
        if self.options['extra_obs_dim'] > 0:
            low.append(0)
            high.append(0)
        low = np.hstack(low)
        high = np.hstack(high)
        obs_space = spaces.Box(low=low, high=high, dtype=np.float64)

        low = -self.options['maxabs_desired_joint_angle']*np.ones(2)
        high = self.options['maxabs_desired_joint_angle']*np.ones(2)
        act_space = spaces.Box(low=low, high=high, dtype=np.float64)

        return (obs_space, act_space)

    def save_options(self, filename):
        with open(f'{filename}_interface_options.json', 'w') as outfile:
            json.dump(self.options, outfile, indent=4, sort_keys=True)

    def update_reward(self):
        # get a copy of both current and prior observation (one inner time step ago)
        (cur_obs, old_obs) = self.obs_history_buffer.get(taps=[0, 1])

        # get data from current observation
        foot_force_x = cur_obs['foot_force'][0]
        foot_force_z = cur_obs['foot_force'][2]
        foot_in_contact = cur_obs['foot_in_contact']
        foot_pos_x = cur_obs['foot_position'][0]
        foot_pos_z = cur_obs['foot_position'][2]
        hip_pos_z = cur_obs['hip_position'][2]
        knee_pos_z = cur_obs['knee_position'][2]
        (theta_hip, theta_knee, omega_hip, omega_knee) = cur_obs['joint_state'].tolist()
        (tau_hip, tau_knee) = cur_obs['joint_torque'].tolist()
        leg_vel_z = cur_obs['leg_vel'][2]

        # get data from prior observation (one inner time step ago)
        (tau_hip_old, tau_knee_old) = old_obs['joint_torque'].tolist()

        if RCRD_AND_DELAY_SELF_JUMPSTATE:
            jump_state = cur_obs['jump_state']
        else:
            jump_state = self.jump_state
        reward = {}
        if self.options['stand_reward'] == 'simplified':
            reward = {}
            mod_theta_err = lambda e_theta: np.abs(((e_theta + np.pi) % (2 * np.pi)) - np.pi)
            reward['main'] = -(mod_theta_err(theta_hip + np.pi / 4) + mod_theta_err(theta_knee + np.pi / 2))
            reward['omega'] = (np.abs(omega_hip) + np.abs(omega_knee)) * -0.08
            reward['foot_x'] = (np.abs(foot_pos_x) * float(knee_pos_z < 0.2)) * -10.
            reward['foot_f_z'] = np.abs(foot_force_z - 7.6) * float(foot_force_z < 7.6) * float(
                self.has_touched_ground) * -.5
            reward['foot_z'] = np.abs(foot_pos_z) * float(self.has_touched_ground) * -10.
            reward['knee_z'] = np.abs(0.1 - knee_pos_z) * float(self.has_touched_ground) * -15.
            if self.options.get('add_torque_penalty_on_air', False):
                reward['tau_knee'] = (np.abs(tau_knee)**2) * (1. - float(self.has_touched_ground)) * -4.

            reward['action_smoothness'] = 0.
            if self.options['add_act_sm_r']:
                done_inner_steps = round(self.options['time_before_reset'] / self.inner_step_time)
                obs_delay_steps = round(self.options['observation_delay'] / self.inner_step_time)

                if self.inner_step_count == 0:
                    self.traj_tau_hip = np.zeros((done_inner_steps), np.float64)
                    self.traj_tau_knee = np.zeros((done_inner_steps), np.float64)
                self.traj_tau_hip[self.inner_step_count] = tau_hip
                self.traj_tau_knee[self.inner_step_count] = tau_knee

                if (self.inner_step_count == (done_inner_steps + obs_delay_steps - 21)):
                    traj_tau_hip = self.traj_tau_hip[obs_delay_steps:(self.inner_step_count+1)]
                    traj_tau_knee = self.traj_tau_knee[obs_delay_steps:(self.inner_step_count+1)]
                    def roughness(traj_tau, depth=0):
                        if traj_tau.size < 5:
                            return 0
                        a = np.abs(np.diff(traj_tau)).sum()
                        b = [0, traj_tau.argmin(), traj_tau.argmax(), traj_tau.size-1]
                        b = sorted(b)
                        if depth > 0:
                            ll_outs = [roughness(traj_tau[i:j], depth=depth-1) for i,j in zip(b[:-1], b[1:])]
                            out = np.sum(ll_outs)
                        else:
                            c = [traj_tau[x] for x in b]
                            d = np.abs(np.diff(c)).sum()
                            out = d - a
                        return out
                    reward['action_smoothness'] += roughness(traj_tau_hip, depth=1)
                    reward['action_smoothness'] += roughness(traj_tau_knee, depth=1)
                    reward['action_smoothness'] *= 100.0

            #Reset history stuff
            if self.options.get('add_omega_smoothness_reward', False):
                reward['omega_smoothness'] = 0.
                if not self.time_greater_equal(0.05):
                    self.all_hist = {}
                    self.all_hist['last_has_touched_ground'] = False
                    self.all_hist['computed_smoothness_reward'] = False

                if (self.all_hist['last_has_touched_ground'] == False) and self.has_touched_ground:
                    self.all_hist['touchdown_time'] = self.get_time()
                    self.all_hist['hip_omegas_touched'] = []
                    self.all_hist['knee_omegas_touched'] = []

                if self.has_touched_ground:
                    required_smooth_window = 0.045
                    if self.get_time() < self.all_hist['touchdown_time'] + required_smooth_window:
                        self.all_hist['hip_omegas_touched'].append(omega_hip)
                        self.all_hist['knee_omegas_touched'].append(omega_knee)
                    elif not self.all_hist['computed_smoothness_reward']:

                        while len(self.all_hist['hip_omegas_touched'])>2:
                            if self.all_hist['hip_omegas_touched'][0] > self.all_hist['hip_omegas_touched'][1]:
                                self.all_hist['hip_omegas_touched'] = self.all_hist['hip_omegas_touched'][1:]
                            else:
                                break
                        hip_om_dist = self.all_hist['hip_omegas_touched'][-1] - self.all_hist['hip_omegas_touched'][0]
                        hip_om_travel = np.abs(np.diff(self.all_hist['hip_omegas_touched'])).sum()
                        hip_om_osc = hip_om_travel - hip_om_dist

                        while len(self.all_hist['knee_omegas_touched'])>2:
                            if self.all_hist['knee_omegas_touched'][0] > self.all_hist['knee_omegas_touched'][1]:
                                self.all_hist['knee_omegas_touched'] = self.all_hist['knee_omegas_touched'][1:]
                            else:
                                #assert self.all_hist['knee_omegas_touched'][0] < -25
                                break
                        knee_om_dist = self.all_hist['knee_omegas_touched'][-1] - self.all_hist['knee_omegas_touched'][0]
                        knee_om_travel = np.abs(np.diff(self.all_hist['knee_omegas_touched'])).sum()
                        knee_om_osc = knee_om_travel - knee_om_dist

                        self.all_hist['hip_om_osc'] = np.abs(hip_om_osc)
                        self.all_hist['knee_om_osc'] = np.abs(knee_om_osc)

                        self.all_hist['computed_smoothness_reward'] = True

                    if (0.2 + required_smooth_window) > (self.get_time() - self.all_hist['touchdown_time']) >=  required_smooth_window:
                        reward['omega_smoothness'] = (self.all_hist['knee_om_osc']) * -0.15

                    self.all_hist['last_has_touched_ground'] = self.has_touched_ground
        else:
            # dense/sparse rewards
            reward['torque_smoothness'] = -self.options['torque_smoothness_coeff'] * ((tau_hip - tau_hip_old)**2 + (tau_knee - tau_knee_old)**2)
            if (jump_state == 0 and self.options['stand_reward'] == 'dense') or (jump_state == 5):    # stand
                reward['foot_pos_x'] = - 1000 * foot_pos_x**2
                reward['posture'] = - 100 * ((hip_pos_z - foot_pos_z) - self.options['posture_height'])**2
                reward['torque'] = - 0.0001 * tau_hip**2 - 0.0001 * tau_knee**2
                reward['velocity'] = - 0.1 * omega_hip**2 - 0.1 * omega_knee**2
                # FIXME: penalize horizontal reaction force on slider
            elif jump_state == 0 and self.options['stand_reward'] == 'sparse':    # stand
                if self.time_greater_equal(0.4):
                    reward['hip_pos_z'] = np.exp(-np.abs(hip_pos_z - self.options['posture_height'])/0.05)
                    reward['foot_pos_z'] = 10*np.exp(-max(0.02, foot_pos_z)/0.01)
                if self.options['a_smooth_coeff'] is not None and self.a_old is not None:
                    reward['a_smooth_hip'] = -self.options['a_smooth_coeff'] * (self.a[0] - self.a_old[0])**2
                    reward['a_smooth_knee'] = -self.options['a_smooth_coeff'] * (self.a[1] - self.a_old[1])**2
                if self.options['a_osc_coeff'] is not None and self.a_old is not None and self.a_old2 is not None:
                    reward['a_osc_hip'] = -self.options['a_osc_coeff'] * max(0, (self.a_old2[0] - self.a_old[0]) * (self.a[0] - self.a_old[0]))
                    reward['a_osc_knee'] = -self.options['a_osc_coeff'] * max(0, (self.a_old2[1] - self.a_old[1]) * (self.a[1] - self.a_old[1]))
                if not self.options['turn_off_constraints']:
                    if self.time_greater_equal(0.25):
                        if np.abs(foot_pos_x) > 0.1:
                            reward['cons_foot_pos_x_hard'] = -self.options['constraint_penalty']
            elif jump_state == 1 and self.options['timed_jump']:
                if leg_vel_z < self.options['max_leg_vel']:
                    reward['jump_vel'] = max(0, self.options['jump_vel_coeff'] * (leg_vel_z - self.leg_vel_peak))
                else:
                    reward['jump_vel'] = -1 * (leg_vel_z - self.options['max_leg_vel'])
                self.leg_vel_peak = max(self.leg_vel_peak, leg_vel_z)
            elif jump_state == 1:  # jump
                reward['hip_pos_z'] = 100 * (hip_pos_z - self.options['max_reach'])
                if self.options['jump_vel_reward']:
                    if leg_vel_z < self.options['max_leg_vel']:
                        reward['jump_vel'] = max(0, 100 * leg_vel_z)
                    else:
                        reward['jump_vel'] = -max(0, 100 * leg_vel_z)
            elif jump_state == 2:  # liftoff
                if self.options['jump_vel_reward']:
                    if leg_vel_z < self.options['max_leg_vel']:
                        reward['jump_vel_liftoff'] = 100000 * leg_vel_z
                    else:
                        reward['jump_vel_liftoff'] = -100000 * leg_vel_z
            elif jump_state == 3:  # in air, far from ground
                reward['hip_pos_z'] = 100 * (hip_pos_z - self.options['max_reach'])
                if self.options['jump_vel_reward']:
                    if leg_vel_z < self.options['max_leg_vel']:
                        reward['jump_vel'] = max(0, 100 * leg_vel_z)
                    else:
                        reward['jump_vel'] = -max(0, 100 * leg_vel_z)
            elif jump_state == 4:  # in air, close to ground
                pass
            elif jump_state == 5:  # in air, for timed period
                pass
            elif jump_state == 6:  # terminate
                pass
            else:
                raise Exception(f'unrecognized jump_state: {jump_state}')

            if not self.options['turn_off_constraints']:
                if np.abs(omega_hip) > self.options['omega_hip_maxabs']:
                    reward = {'cons_omega_hip': -self.options['constraint_penalty']}
                if np.abs(omega_knee) > self.options['omega_knee_maxabs']:
                    reward = {'cons_omega_knee': -self.options['constraint_penalty']}
                if not (self.options['theta_hip_bounds'][0] <= theta_hip <= self.options['theta_hip_bounds'][1]):
                    reward = {'cons_theta_hip': -self.options['constraint_penalty']}
                if not (self.options['theta_knee_bounds'][0] <= theta_knee <= self.options['theta_knee_bounds'][1]):
                    reward = {'cons_theta_knee': -self.options['constraint_penalty']}
                if hip_pos_z < self.options['hip_minimum_z']:
                    reward = {'cons_hip_z': -self.options['constraint_penalty']}
                if knee_pos_z < self.options['knee_minimum_z']:
                    reward = {'cons_knee_z': -self.options['constraint_penalty']}
                # if self.has_touched_ground and not foot_in_contact and self.jump_state == 0:
                #     reward = {'cons_stay_on_ground': -self.options['constraint_penalty']}

            if self.simulation_error:
                reward = {'sim_error': -self.options['constraint_penalty']}

        r = 0.0
        for key in reward:
            r += reward[key]

        self.r_inner = r
        self.reward = reward

        if Debug_reward:
            print(f"Rstep %d:" % self.inner_step_count)
            print(f'  -->    theta                = %+05.8f, %+05.8f' % tuple(self.observation["joint_state"][0:2]))
            print(f'  -->    omega                = %+05.8f, %+05.8f' % tuple(self.observation["joint_state"][2:4]))
            print('\n')
            print(f'  -->    foot_pos_x,z         = %+05.8f, %+05.8f' % tuple(self.observation["foot_position"][0::2]))
            print(f'  -->    knee_pos_z,z         = %+05.8f, %+05.8f' % tuple(self.observation["knee_position"][0::2]))
            print(f'  -->    tau                  = %+05.8f, %+05.8f' % tuple(cur_obs['joint_torque']))
            print(f'  -->    tau_old              = %+05.8f, %+05.8f' % tuple(old_obs['joint_torque']))
            print(f'  -->    jump_state           = %d' % jump_state)
            for ii_, (key, val) in enumerate(self.reward.items()):
                print(f"  --> R{ii_+1}) Reward['{key}'] {(20-len(str(key))) * ' '}= %+05.8f" % val)
            print(f"  --> R*) Total Reward                   = %+05.8f" % self.r_inner)

class GymEnv:
    def __init__(self, sim_interface):
        self.sim_interface = sim_interface
        (self.observation_dim, self.action_dim) = self.sim_interface.get_dims()
        if has_gym:
            # Need these to match OpenAI Gym standard
            (self.observation_space, self.action_space) = self.sim_interface.get_specs()
        self.s = self.reset()

    def get_timestep(self):
        return self.sim_interface.get_timestep()

    def step(self, a):
        s, r, done = self.sim_interface.step(a)
        self.s = s
        return (self.s, r, done, {})

    def reset(self, np_random=np.random):
        self.s = self.sim_interface.reset(np_random=np_random)
        return self.s

    def save(self, filename):
        self.sim_interface.save_options(filename)

    def get_options(self):
        return self.sim_interface.get_options()

    def get_nondefault_options(self):
        return self.sim_interface.get_nondefault_options()

    def get_metadata(self):
        return self.sim_interface.get_metadata()

    def simulate_and_plot(self, action_greedy, disable_randomizable=False,
            write_snapshot=False, logdir_base=None, n_steps=None, r_avg=None,
            log_number=None):
        # FIXME - you should be more careful about clearing memory. The interface
        # has a copy of the sim_env. In theory, the interface could use information
        # from sim_env to initialize the memory. So, you'd want to clear memory
        # after calling sim_env.reset(). However, sim_env.reset() is called in
        # simulate_and_plot, which doesn't know anything about the interface. We
        # will ignore this for now.
        return simulate_and_plot(self.sim_interface, action_greedy,
                disable_randomizable=disable_randomizable,
                write_snapshot=write_snapshot,
                logdir_base=logdir_base,
                n_steps=n_steps, r_avg=r_avg,
                log_number=log_number)

def test_change_model():
    sim_task = SimTask(random=None)
    xml_string = open('leg.xml').read()
    sim_env = SimEnv(sim_task, xml_string=xml_string)
    mujoco.wrapper.save_last_parsed_model_to_xml('test_change_model_original.xml')
    sim_env.set_ground_contact('noncompliant')
    sim_env.set_upperleg_length(0.2)
    sim_env.set_lowerleg_length(0.12)
    sim_env.set_body_mass('upperleg', 0.15)
    sim_env.set_body_mass('lowerleg', 0.05)
    sim_env.set_body_inertia('upperleg', np.array([0.3, 0.2, 0.1]))
    sim_env.set_body_inertia('lowerleg', np.array([0.1, 0.2, 0.3]))
    sim_env.set_body_com('upperleg', np.array([0, 0, -0.05]))
    sim_env.set_body_com('lowerleg', np.array([0, 0, -0.09]))
    sim_env.set_joint_frictionloss('slider', 0.1)
    sim_env.set_joint_frictionloss('hip', 0.2)
    sim_env.set_joint_frictionloss('knee', 0.3)
    sim_env.set_joint_damping('slider', 0.03)
    sim_env.set_joint_damping('hip', 0.04)
    sim_env.set_joint_damping('knee', 0.05)
    sim_env.set_joint_armature('hip', 1e-6)
    sim_env.set_joint_armature('knee', 2e-6)
    mujoco.wrapper.save_last_parsed_model_to_xml('test_change_model_modified.xml')

def main():
    test_change_model()


def Rx(h): # rotation about x axis by angle h
    return np.array([[1, 0, 0], [0, np.cos(h), -np.sin(h)], [0, np.sin(h), np.cos(h)]])

def Ry(h): # rotation about y axis by angle h
    return np.array([[np.cos(h), 0, np.sin(h)], [0, 1, 0], [-np.sin(h), 0, np.cos(h)]])

def Rz(h): # rotation about z axis by angle h
    return np.array([[np.cos(h), -np.sin(h), 0], [np.sin(h), np.cos(h), 0], [0, 0, 1]])

def dRx(h): # derivative of Rx with respect to h
    return np.array([[0, 0, 0], [0, -np.sin(h), -np.cos(h)], [0, np.cos(h), -np.sin(h)]])

def dRy(h): # derivative of Ry with respect to h
    return np.array([[-np.sin(h), 0, np.cos(h)], [0, 0, 0], [-np.cos(h), 0, -np.sin(h)]])

def dRz(h): # derivative of Rz with respect to h
    return np.array([[-np.sin(h), -np.cos(h), 0], [np.cos(h), -np.sin(h), 0], [0, 0, 0]])

def geometry_params():
    """
    Geometry vectors in the reference configuration (leg stretched out along the x axis)
    """
    r1 = np.array([0, -0.04, 0]) # offset at hip
    r2 = np.array([0.14, 0, 0]) # upper leg in horizontal position
    r3 = np.array([0.14, 0, 0]) # lower leg in horizontal position
    return (r1, r2, r3)


def foot_position(q):
    """
    Foot position p as a function of angles q = (hip_side_angle, hip_angle, knee_angle)
    """
    (r1, r2, r3) = geometry_params()
    p = Rx(q[0]) @ (r1 + Ry(-q[1]) @ (r2 + Ry(-q[2]) @ r3))
    return p


def foot_p_and_v(q, dq):
    p = foot_position(q)
    J = jacobian(q)
    v = J @ dq
    return p, v


def jacobian(q):
    """
    Derivative dp/dq of foot position with respect to angles.
    """
    (r1, r2, r3) = geometry_params()
    J0 = dRx(q[0]) @ (r1 + Ry(-q[1]) @ (r2 + Ry(-q[2]) @ r3))
    J1 = - Rx(q[0]) @ dRy(-q[1]) @ (r2 + Ry(-q[2]) @ r3)
    J2 = - Rx(q[0]) @ Ry(-q[1]) @ dRy(-q[2]) @ r3
    J = np.stack((J0, J1, J2), axis=1)
    return J


def joint_state_to_angles(joint_state):
    """
    (q, dq) = obs_to_angles(obs)

    joint_state is (theta_hip, theta_knee, omega_hip, omega_knee)
    q and dq are the angles and angular velocities
    """
    (uh, lh, uw, lw) = joint_state
    q = np.array([0, uh, lh])
    dq = np.array([0, uw, lw])
    return (q, dq)

def pd_action(obs, pd=np.array([0, -0.04, -0.15]), vd=np.array([0, 0, 0]), kp=400, kd=3, use_viscous_comp=False):
    """
    a = pd_action(obs)

    obs is the dmc_env observation vector
    a is the dmc_env action vector
    pd is the desired foot position, relative to the base
    vd is the desired foot velocity, relative to the base
    kp is the position gain
    kd is the velocity gain
    use_viscous_comp is whether to add the viscous friction compensation term to the torques
    """
    q, dq = obs_to_angles(obs)
    return pd_action_from_q(q, dq, pd, vd, kp, kd)


def pd_action_from_q(q, dq, pd=np.array([0, -0.04, -0.15]), vd=np.array([0, 0, 0]), kp=400, kd=3, use_viscous_comp=False):
    """
    a = pd_action(q, dq)

    q is the angle 3-vector
    dq is the angular velocity 3-vector
    a is the dmc_env action vector
    pd is the desired foot position, relative to the base
    vd is the desired foot velocity, relative to the base
    kp is the position gain
    kd is the velocity gain
    use_viscous_comp is whether to add the viscous friction compensation term to the torques
    """
    p = foot_position(q)
    J = jacobian(q)
    v = J @ dq

    Kp = kp * np.eye(3)
    Kd = kd * np.eye(3)
    F_P = Kp @ (pd - p)
    F_D = Kd @ (vd - v)
    F_PD = F_P + F_D

    tau = J.T @ F_PD
    if use_viscous_comp:
        tau += np.diag([0.006, 0.006, 0.02]) @ dq
    return np.array([tau[1], tau[2]])


def pos_and_vel(theta, omega, hip_pos_z=None, hip_vel_z=None):
    """theta is Nx2, omega is Nx2, hip_pos_z is N, hip_vel_z is N"""
    r1,r2,r3 = geometry_params()
    L2 = r2[0]
    L2C = L2/2
    L3 = r3[0]
    L3C = L3/2

    # relative to hip
    COM2_pos_x = L2C*np.cos(theta[:,0])
    COM2_pos_z = L2C*np.sin(theta[:,0])
    knee_pos_x = L2*np.cos(theta[:,0])
    knee_pos_z = L2*np.sin(theta[:,0])
    COM3_pos_x = knee_pos_x + L3C*np.cos(theta[:,0] + theta[:,1])
    COM3_pos_z = knee_pos_z + L3C*np.sin(theta[:,0] + theta[:,1])
    foot_pos_x = knee_pos_x + L3*np.cos(theta[:,0] + theta[:,1])
    foot_pos_z = knee_pos_z + L3*np.sin(theta[:,0] + theta[:,1])

    # relative to hip
    COM2_vel_x = -L2C*np.sin(theta[:,0]) * omega[:,0]
    COM2_vel_z = L2C*np.cos(theta[:,0]) * omega[:,0]
    knee_vel_x = -L2*np.sin(theta[:,0]) * omega[:,0]
    knee_vel_z = L2*np.cos(theta[:,0]) * omega[:,0]
    COM3_vel_x = knee_vel_x - L3C*np.sin(theta[:,0] + theta[:,1]) * (omega[:,0] + omega[:,1])
    COM3_vel_z = knee_vel_z + L3C*np.cos(theta[:,0] + theta[:,1]) * (omega[:,0] + omega[:,1])
    foot_vel_x = knee_vel_x - L3*np.sin(theta[:,0] + theta[:,1]) * (omega[:,0] + omega[:,1])
    foot_vel_z = knee_vel_z + L3*np.cos(theta[:,0] + theta[:,1]) * (omega[:,0] + omega[:,1])

    # if we don't have hip data, assume the foot is on the ground
    if hip_pos_z is None:
        hip_pos_z = -foot_pos_z.copy()
    if hip_vel_z is None:
        hip_vel_z = -foot_vel_z.copy()

    # fix up data with hip offset
    COM2_pos_z += hip_pos_z
    knee_pos_z += hip_pos_z
    COM3_pos_z += hip_pos_z
    foot_pos_z += hip_pos_z

    # fix up data with hip offset
    COM2_vel_z += hip_vel_z
    knee_vel_z += hip_vel_z
    COM3_vel_z += hip_vel_z
    foot_vel_z += hip_vel_z

    # we know the hip always has x=0
    hip_pos_x = np.zeros_like(hip_pos_z)
    hip_vel_x = np.zeros_like(hip_pos_z)

    return (
        hip_pos_x,
        hip_pos_z,
        COM2_pos_x,
        COM2_pos_z,
        knee_pos_x,
        knee_pos_z,
        COM3_pos_x,
        COM3_pos_z,
        foot_pos_x,
        foot_pos_z,
        hip_vel_x,
        hip_vel_z,
        COM2_vel_x,
        COM2_vel_z,
        knee_vel_x,
        knee_vel_z,
        COM3_vel_x,
        COM3_vel_z,
        foot_vel_x,
        foot_vel_z,
    )

def energy(theta, omega, hip_pos_z=None, hip_vel_z=None):
    """theta is Nx2, omega is Nx2, hip_pos_z is N, hip_vel_z is N"""
    g = 9.81
    mB = 0.59
    mU = 0.09
    mL = 0.05
    IB = 0.000078
    IU = 0.0003
    IL = 0.00015

    (
        hip_pos_x,
        hip_pos_z,
        COM2_pos_x,
        COM2_pos_z,
        knee_pos_x,
        knee_pos_z,
        COM3_pos_x,
        COM3_pos_z,
        foot_pos_x,
        foot_pos_z,
        hip_vel_x,
        hip_vel_z,
        COM2_vel_x,
        COM2_vel_z,
        knee_vel_x,
        knee_vel_z,
        COM3_vel_x,
        COM3_vel_z,
        foot_vel_x,
        foot_vel_z,
    ) = pos_and_vel(theta, omega, hip_pos_z, hip_vel_z)

    base_KE = 0.5*mB*(hip_vel_x**2 + hip_vel_z**2) + 0.5*IB*omega[:,0]**2
    upper_KE = 0.5*mU*(COM2_vel_x**2 + COM2_vel_z**2) + 0.5*IU*omega[:,0]**2
    lower_KE = 0.5*mL*(COM3_vel_x**2 + COM3_vel_z**2) + 0.5*IL*(omega[:,0] + omega[:,1])**2

    base_PE = mB*g*hip_pos_z
    upper_PE = mU*g*COM2_pos_z
    lower_PE = mL*g*COM3_pos_z

    KE = base_KE + upper_KE + lower_KE
    PE = base_PE + upper_PE + lower_PE

    E = KE + PE

    return (E, KE, PE, base_KE, upper_KE, lower_KE, base_PE, upper_PE, lower_PE,
        hip_pos_x,
        hip_pos_z,
        COM2_pos_x,
        COM2_pos_z,
        knee_pos_x,
        knee_pos_z,
        COM3_pos_x,
        COM3_pos_z,
        foot_pos_x,
        foot_pos_z,
        hip_vel_x,
        hip_vel_z,
        COM2_vel_x,
        COM2_vel_z,
        knee_vel_x,
        knee_vel_z,
        COM3_vel_x,
        COM3_vel_z,
        foot_vel_x,
        foot_vel_z,
            )

def simulate_and_plot(sim_interface, policy, fig_width=6,
        fig_height_per_plot=2, exp_data=None, exp_step_range=None, sim_s0=0,
        force_baseline_s=None, return_fig=True, return_traj=False,
        plot_raw=False, plot_true=False, exp_eng_s0=None, exp_eng_s1=None,
        disable_randomizable=False, write_snapshot=False, logdir_base=None,
        n_steps=None, r_avg=None, log_number=None,
        do_animation=False, fps=50, video_name='video'):
    """
    policy must accept s and return a
    """

    traj = {}

    def add_data(key, value):
        if key not in traj:
            traj[key] = []
        traj[key].append(value)

    def update(data_buffer):
        for data in data_buffer:
            obs = data['observation']
            add_data('joint_state_raw', data['joint_state_raw'].copy())
            add_data('joint_state', data['joint_state'].copy())
            add_data('hip_pos', obs['hip_position'].copy())
            add_data('knee_pos', obs['knee_position'].copy())
            add_data('foot_pos', obs['foot_position'].copy())
            add_data('hip_vel', obs['hip_velocity'].copy())
            add_data('knee_vel', obs['knee_velocity'].copy())
            add_data('foot_vel', obs['foot_velocity'].copy())
            add_data('leg_pos', obs['leg_pos'].copy())
            add_data('leg_vel', obs['leg_vel'].copy())
            add_data('foot_offset', obs['foot_offset'].copy())
            add_data('foot_offset_vel', obs['foot_offset_vel'].copy())
            add_data('foot_offset_true', obs['foot_offset_true'].copy())
            add_data('foot_offset_vel_true', obs['foot_offset_vel_true'].copy())
            add_data('foot_offset_raw', obs['foot_offset_raw'].copy())
            add_data('foot_offset_vel_raw', obs['foot_offset_vel_raw'].copy())
            add_data('foot_in_contact', obs['foot_in_contact'])
            add_data('is_jumping', data['is_jumping'])
            add_data('a', data['a'].copy())
            add_data('a_raw', data['a_raw'].copy())
            add_data('desired_foot_offset', data['desired_foot_offset'].copy())
            add_data('desired_joint_angle', data['desired_joint_angle'].copy())
            add_data('torque_components', data['torque_components'].copy())
            add_data('joint_torque', obs['joint_torque'].copy())
            add_data('joint_torque_command', data['joint_torque_command'].copy())
            add_data('foot_force', obs['foot_force'].copy())
            add_data('total_reward', data['r_inner'])
            add_data('reward', data['reward'])

    if do_animation:
        import cv2

        def grabMujocoFrame(env, height=600, width=480, camera_id='fixed-cam'):
            rgbArr = env.physics.render(height, width, camera_id=camera_id)
            return cv2.cvtColor(rgbArr, cv2.COLOR_BGR2RGB)

        frame = grabMujocoFrame(sim_interface.sim_env)
        height, width, layers = frame.shape
        video = cv2.VideoWriter(video_name + '.mp4',
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps, (width, height))

    s = sim_interface.reset(disable_randomizable=disable_randomizable)
    update(sim_interface.data_buffer)
    done = False
    while not done:
        a = policy(s)
        s, r, done = sim_interface.step(a)
        update(sim_interface.data_buffer)
        if do_animation:
            frame = grabMujocoFrame(sim_interface.sim_env)
            video.write(frame)
    if do_animation:
        video.release()

    # unpack and fill the reward components
    max_r_jump_vel = 0.0
    reward_keys = sorted(set().union(*traj['reward']))
    for key in reward_keys:
        traj['r_' + key] = []
    for cur_reward in traj['reward']:
        for key in reward_keys:
            traj['r_' + key].append(cur_reward.get(key, 0))

    for key in traj:
        traj[key] = np.array(traj[key])
        if key == 'r_jump_vel':
            max_r_jump_vel = np.max(traj[key])

    # if we have experimental data
    if exp_step_range is not None:
        s0 = exp_step_range[0]
        s1 = exp_step_range[1]
    elif exp_data is not None:
        s0 = 0
        s1 = len(exp_data['q'])

    t_sim = np.arange(traj['joint_state'].shape[0] - sim_s0) / 4000
    if exp_data is not None:
        t_exp = np.arange(s1 - s0) / 4000

    if force_baseline_s is not None:
        force_baseline = exp_data['force'][force_baseline_s,:]
    else:
        force_baseline = np.zeros(3)

    interface_options = sim_interface.get_options()
    torque_delay_steps = int(np.round(interface_options['torque_delay'] * interface_options['inner_loop_rate']))
    observation_delay_steps = int(np.round(interface_options['observation_delay'] * interface_options['inner_loop_rate']))

    (
        traj['E'],
        traj['KE'],
        traj['PE'],
        traj['base_KE'],
        traj['upper_KE'],
        traj['lower_KE'],
        traj['base_PE'],
        traj['upper_PE'],
        traj['lower_PE'],
        traj['eng_hip_pos_x'],
        traj['eng_hip_pos_z'],
        traj['eng_COM2_pos_x'],
        traj['eng_COM2_pos_z'],
        traj['eng_knee_pos_x'],
        traj['eng_knee_pos_z'],
        traj['eng_COM3_pos_x'],
        traj['eng_COM3_pos_z'],
        traj['eng_foot_pos_x'],
        traj['eng_foot_pos_z'],
        traj['eng_hip_vel_x'],
        traj['eng_hip_vel_z'],
        traj['eng_COM2_vel_x'],
        traj['eng_COM2_vel_z'],
        traj['eng_knee_vel_x'],
        traj['eng_knee_vel_z'],
        traj['eng_COM3_vel_x'],
        traj['eng_COM3_vel_z'],
        traj['eng_foot_vel_x'],
        traj['eng_foot_vel_z'],
    ) = energy(traj['joint_state'][:,:2], traj['joint_state'][:,2:], traj['hip_pos'][:,2], traj['hip_vel'][:,2])

    if exp_data is not None:
        (
            exp_data['E'],
            exp_data['KE'],
            exp_data['PE'],
            exp_data['base_KE'],
            exp_data['upper_KE'],
            exp_data['lower_KE'],
            exp_data['base_PE'],
            exp_data['upper_PE'],
            exp_data['lower_PE'],
            exp_data['eng_hip_pos_x'],
            exp_data['eng_hip_pos_z'],
            exp_data['eng_COM2_pos_x'],
            exp_data['eng_COM2_pos_z'],
            exp_data['eng_knee_pos_x'],
            exp_data['eng_knee_pos_z'],
            exp_data['eng_COM3_pos_x'],
            exp_data['eng_COM3_pos_z'],
            exp_data['eng_foot_pos_x'],
            exp_data['eng_foot_pos_z'],
            exp_data['eng_hip_vel_x'],
            exp_data['eng_hip_vel_z'],
            exp_data['eng_COM2_vel_x'],
            exp_data['eng_COM2_vel_z'],
            exp_data['eng_knee_vel_x'],
            exp_data['eng_knee_vel_z'],
            exp_data['eng_COM3_vel_x'],
            exp_data['eng_COM3_vel_z'],
            exp_data['eng_foot_vel_x'],
            exp_data['eng_foot_vel_z'],
        ) = energy(np.roll(exp_data['q'][:,1:], -observation_delay_steps, axis=0), np.roll(exp_data['dq'][:,1:], -observation_delay_steps, axis=0))

    incr_work_sim = traj['joint_torque'] * (np.roll(traj['joint_state'][:,:2], -1, axis=0) - traj['joint_state'][:,:2])
    traj['power'] = incr_work_sim.sum(axis=1) / interface_options['inner_loop_rate']
    traj['power_hip'] = incr_work_sim[:,0] / interface_options['inner_loop_rate']
    traj['power_knee'] = incr_work_sim[:,1] / interface_options['inner_loop_rate']
    work_sim = np.cumsum(incr_work_sim, axis=0)
    work_sim -= work_sim[sim_s0,:]
    traj['work'] = work_sim.sum(axis=1)
    traj['work_hip'] = work_sim[:,0]
    traj['work_knee'] = work_sim[:,1]
    traj['energy_loss'] = traj['work'] - (traj['E'] - traj['E'][sim_s0])

    if exp_data is not None:
        exp_torque = np.roll(exp_data['tau_Nm'][:,1:], torque_delay_steps, axis=0)
        exp_theta = np.roll(exp_data['q'][:,1:], -observation_delay_steps, axis=0)
        exp_omega = np.roll(exp_data['dq'][:,1:], -observation_delay_steps, axis=0)

        abs_omega_hip = np.abs(exp_omega[:,0])
        abs_omega_knee = np.abs(exp_omega[:,1])
        if False: # old motor saturation model
            maxabs_tau_hip = interface_options['maxabs_tau']
            maxabs_tau_knee = interface_options['maxabs_tau']

            maxabs_omega_for_tau = 21.66 #30.79
            tau_omega_slope = -0.397

            maxabs_tau_hip = np.maximum(0, maxabs_tau_hip + np.minimum(0, tau_omega_slope * (abs_omega_hip - maxabs_omega_for_tau)))
            maxabs_tau_knee = np.maximum(0, maxabs_tau_knee + np.minimum(0, tau_omega_slope * (abs_omega_knee - maxabs_omega_for_tau)))

            exp_torque[:,0] = np.clip(exp_torque[:,0], -maxabs_tau_hip, maxabs_tau_hip)
            exp_torque[:,1] = np.clip(exp_torque[:,1], -maxabs_tau_knee, maxabs_tau_knee)
        if True: # dyno motor saturation model
            for i in range(exp_torque.shape[0]):
                exp_torque[i,0] = dyno_motor_model(exp_torque[i,0], abs_omega_hip[i])
                exp_torque[i,1] = dyno_motor_model(exp_torque[i,1], abs_omega_knee[i])

        incr_work_exp = interface_options['output_torque_scale'] * exp_torque * (np.roll(exp_theta, -1, axis=0) - exp_theta)
        exp_data['power'] = incr_work_exp.sum(axis=1) / interface_options['inner_loop_rate']
        exp_data['power_hip'] = incr_work_exp[:,0] / interface_options['inner_loop_rate']
        exp_data['power_knee'] = incr_work_exp[:,1] / interface_options['inner_loop_rate']
        work_exp = np.cumsum(incr_work_exp, axis=0)
        work_exp -= work_exp[s0,:]
        exp_data['work'] = work_exp.sum(axis=1)
        exp_data['work_hip'] = work_exp[:,0]
        exp_data['work_knee'] = work_exp[:,1]
        exp_data['energy_loss'] = exp_data['work'] - (exp_data['E'] - exp_data['E'][s0])

    if exp_data is not None:
        if exp_eng_s0 is None:
            exp_eng_s0 = s0
        if exp_eng_s1 is None:
            exp_eng_s1 = s1
        t_exp_eng = t_exp[(exp_eng_s0-s0):(exp_eng_s1-s0)]

    n_plots = 24
    if exp_data is not None:
        n_plots += 8
    if plot_true:
        n_plots += 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(fig_width, n_plots * fig_height_per_plot), sharex=True)

    i_axis = 0
    ax = axes[i_axis]
    max_hip_height = 0.0
    # max_hip_height = np.amax(traj['hip_pos'][sim_s0:,2])
    max_hip_height = np.amax(traj['hip_pos'][sim_s0+4000:,2]) # Ignore the initial drop
    for key in reward_keys:
        ax.plot(t_sim, traj['r_' + key][sim_s0:], linewidth=1, label=f'r_{key}')
    ax.legend(bbox_to_anchor=(0., 1.2, 1., .2), loc='lower left',
              ncol=3, mode="expand", borderaxespad=0.)
    snapshot_info = (f'avg_r = {np.mean(traj["total_reward"]):.4},'
        f'max_r_jump_vel = {max_r_jump_vel:.4},'
        f'max_height = {max_hip_height:.4}' )
    ax.set_title(snapshot_info)
    ax.grid()
    ax.set_yscale('symlog')
    # Write training data into a .csv file (or not)
    if write_snapshot:
        save_header = 'r_avg_snapshoot,max_r_jump_vel,max_height'
        save_data_row = f'{np.mean(traj["total_reward"]):.4},{max_r_jump_vel:.4},{max_hip_height:.4}'
        if n_steps is not None:
            save_header = 'Step,' + save_header
            save_data_row = f'{n_steps},' + save_data_row
        if r_avg is not None:
            save_header += ',r_avg_rollouts'
            save_data_row += f',{r_avg:.4}'
        if log_number is not None:
            save_header = 'Idx,' + save_header
            save_data_row = f'{log_number},' + save_data_row
        write_snapshot_info(snapshot_header=save_header,
                            snapshot_info=save_data_row,
                            logdir_base=logdir_base)

    i_axis += 1
    ax = axes[i_axis]
    ax.plot(t_sim, traj['total_reward'][sim_s0:], 'k-', linewidth=1, label='reward')
    ax.legend(loc='upper right')
    ax.grid()
    ax.set_yscale('symlog')

    # FIXME: scaled wrong
    i_axis += 1
    ax = axes[i_axis]
    ax.plot(t_sim, np.cumsum(traj['total_reward'][sim_s0:]), 'k-', linewidth=1, label='cumulative reward')
    ax.legend(loc='upper right')
    ax.grid()
    ax.set_yscale('symlog')

    # FIXME: scaled wrong
    i_axis += 1
    ax = axes[i_axis]
    for key in reward_keys:
        ax.plot(t_sim, np.cumsum(traj['r_' + key][sim_s0:]), linewidth=1, label=f'cumul r_{key}')
    ax.grid()
    ax.set_yscale('symlog')

    i_axis += 1
    ax = axes[i_axis]
    ax.plot(t_sim, traj['joint_state'][sim_s0:,0], 'r-', linewidth=1, label='theta_hip_sim')
    ax.plot(t_sim, traj['joint_state'][sim_s0:,1], 'b-', linewidth=1, label='theta_knee_sim')
    if plot_raw:
        ax.plot(t_sim, traj['joint_state_raw'][sim_s0:,0], 'r-', alpha=0.2, linewidth=1, label='theta_hip_raw_sim')
        ax.plot(t_sim, traj['joint_state_raw'][sim_s0:,1], 'b-', alpha=0.2, linewidth=1, label='theta_knee_raw_sim')
    if exp_data is not None:
        ax.plot(t_exp, exp_data['q'][s0:s1,1], 'r--', linewidth=1, label='theta_hip_exp')
        ax.plot(t_exp, exp_data['q'][s0:s1,2], 'b--', linewidth=1, label='theta_knee_exp')
        if plot_raw:
            ax.plot(t_exp, exp_data['q_raw'][s0:s1,1], 'r--', alpha=0.2, linewidth=1, label='theta_hip_raw_exp')
            ax.plot(t_exp, exp_data['q_raw'][s0:s1,2], 'b--', alpha=0.2, linewidth=1, label='theta_knee_raw_exp')
    ax.legend(loc='upper right')
    ax.grid()

    i_axis += 1
    ax = axes[i_axis]
    ax.plot(t_sim, traj['joint_state'][sim_s0:,2], 'r-', linewidth=1, label='omega_hip_sim')
    ax.plot(t_sim, traj['joint_state'][sim_s0:,3], 'b-', linewidth=1, label='omega_knee_sim')
    if plot_raw:
        ax.plot(t_sim, traj['joint_state_raw'][sim_s0:,2], 'r-', alpha=0.2, linewidth=1, label='omega_hip_raw_sim')
        ax.plot(t_sim, traj['joint_state_raw'][sim_s0:,3], 'b-', alpha=0.2, linewidth=1, label='omega_knee_raw_sim')
    if exp_data is not None:
        ax.plot(t_exp, exp_data['dq'][s0:s1,1], 'r--', linewidth=1, label='omega_hip_exp')
        ax.plot(t_exp, exp_data['dq'][s0:s1,2], 'b--', linewidth=1, label='omega_knee_exp')
        if plot_raw:
            ax.plot(t_exp, exp_data['dq_raw'][s0:s1,1], 'r--', alpha=0.2, linewidth=1, label='omega_hip_raw_exp')
            ax.plot(t_exp, exp_data['dq_raw'][s0:s1,2], 'b--', alpha=0.2, linewidth=1, label='omega_knee_raw_exp')
    ax.legend(loc='upper right')
    ax.grid()

    if exp_data is not None:
        i_axis += 1
        ax = axes[i_axis]
        ax.plot(t_sim, traj['joint_state'][sim_s0:,2], 'r-', linewidth=1, label='omega_hip_sim')
        ax.plot(t_sim, traj['joint_state'][sim_s0:,3], 'b-', linewidth=1, label='omega_knee_sim')
        if plot_raw:
            ax.plot(t_sim, traj['joint_state_raw'][sim_s0:,2], 'r-', alpha=0.2, linewidth=1, label='omega_hip_raw_sim')
            ax.plot(t_sim, traj['joint_state_raw'][sim_s0:,3], 'b-', alpha=0.2, linewidth=1, label='omega_knee_raw_sim')
        ax.legend(loc='upper right')
        ax.grid()

    if exp_data is not None:
        i_axis += 1
        ax = axes[i_axis]
        ax.plot(t_exp, exp_data['dq'][s0:s1,1], 'r--', linewidth=1, label='omega_hip_exp')
        ax.plot(t_exp, exp_data['dq'][s0:s1,2], 'b--', linewidth=1, label='omega_knee_exp')
        if plot_raw:
            ax.plot(t_exp, exp_data['dq_raw'][s0:s1,1], 'r--', alpha=0.2, linewidth=1, label='omega_hip_raw_exp')
            ax.plot(t_exp, exp_data['dq_raw'][s0:s1,2], 'b--', alpha=0.2, linewidth=1, label='omega_knee_raw_exp')
        ax.legend(loc='upper right')
        ax.grid()

    i_axis += 1
    ax = axes[i_axis]
    ax.plot(t_sim, traj['joint_torque_command'][sim_s0:,0], 'r-', linewidth=1, label='tau_cmd_hip_sim')
    ax.plot(t_sim, traj['joint_torque_command'][sim_s0:,1], 'b-', linewidth=1, label='tau_cmd_knee_sim')
    if exp_data is not None:
        ax.plot(t_exp, exp_data['tau_Nm'][s0:s1,1], 'r--', linewidth=1, label='tau_cmd_hip_exp')
        ax.plot(t_exp, exp_data['tau_Nm'][s0:s1,2], 'b--', linewidth=1, label='tau_cmd_knee_exp')
    ax.legend(loc='upper right')
    ax.grid()

    if exp_data is not None:
        i_axis += 1
        ax = axes[i_axis]
        ax.plot(t_sim, traj['joint_torque_command'][sim_s0:,0], 'r-', linewidth=1, label='tau_cmd_hip_sim')
        ax.plot(t_sim, traj['joint_torque_command'][sim_s0:,1], 'b-', linewidth=1, label='tau_cmd_knee_sim')
        ax.legend(loc='upper right')
        ax.grid()

    if exp_data is not None:
        i_axis += 1
        ax = axes[i_axis]
        ax.plot(t_exp, exp_data['tau_Nm'][s0:s1,1], 'r--', linewidth=1, label='tau_cmd_hip_exp')
        ax.plot(t_exp, exp_data['tau_Nm'][s0:s1,2], 'b--', linewidth=1, label='tau_cmd_knee_exp')
        ax.legend(loc='upper right')
        ax.grid()

    i_axis += 1
    ax = axes[i_axis]
    ax.plot(t_sim, traj['foot_force'][sim_s0:,0], 'r-', linewidth=1, label='foot_force_x_sim')
    ax.plot(t_sim, traj['foot_force'][sim_s0:,1], 'b-', linewidth=1, label='foot_force_y_sim')
    ax.plot(t_sim, traj['foot_force'][sim_s0:,2], 'g-', linewidth=1, label='foot_force_z_sim')
    if exp_data is not None:
        ax.plot(t_exp, -(exp_data['force'][s0:s1,0] - force_baseline[0]), 'r--', linewidth=1, label='foot_force_x_exp')
        ax.plot(t_exp, exp_data['force'][s0:s1,1] - force_baseline[1], 'b--', linewidth=1, label='foot_force_y_exp')
        ax.plot(t_exp, exp_data['force'][s0:s1,2] - force_baseline[2], 'g--', linewidth=1, label='foot_force_z_exp')
    ax.legend(loc='upper right')
    ax.grid()

    if exp_data is not None:
        i_axis += 1
        ax = axes[i_axis]
        ax.plot(t_sim, traj['foot_force'][sim_s0:,0], 'r-', linewidth=1, label='foot_force_x_sim')
        ax.plot(t_sim, traj['foot_force'][sim_s0:,1], 'b-', linewidth=1, label='foot_force_y_sim')
        ax.plot(t_sim, traj['foot_force'][sim_s0:,2], 'g-', linewidth=1, label='foot_force_z_sim')
        ax.legend(loc='upper right')
        ax.grid()

    if exp_data is not None:
        i_axis += 1
        ax = axes[i_axis]
        ax.plot(t_exp, -(exp_data['force'][s0:s1,0] - force_baseline[0]), 'r--', linewidth=1, label='foot_force_x_exp')
        ax.plot(t_exp, exp_data['force'][s0:s1,1] - force_baseline[1], 'b--', linewidth=1, label='foot_force_y_exp')
        ax.plot(t_exp, exp_data['force'][s0:s1,2] - force_baseline[2], 'g--', linewidth=1, label='foot_force_z_exp')
        ax.legend(loc='upper right')
        ax.grid()

    i_axis += 1
    ax = axes[i_axis]
    ax.plot(t_sim, traj['joint_torque'][sim_s0:,0], 'r-', linewidth=1, label='tau_hip_sim')
    ax.plot(t_sim, traj['joint_torque'][sim_s0:,1], 'b-', linewidth=1, label='tau_knee_sim')
    ax.legend(loc='upper right')
    ax.grid()

    i_axis += 1
    ax = axes[i_axis]
    ax.plot(t_sim, traj['torque_components'][sim_s0:,0], 'r-', linewidth=1, label='tau_cmd_hip_p_sim')
    ax.plot(t_sim, traj['torque_components'][sim_s0:,1], 'b-', linewidth=1, label='tau_cmd_knee_p_sim')
    ax.legend(loc='upper right')
    ax.grid()

    i_axis += 1
    ax = axes[i_axis]
    ax.plot(t_sim, traj['torque_components'][sim_s0:,2], 'r-', linewidth=1, label='tau_cmd_hip_d_sim')
    ax.plot(t_sim, traj['torque_components'][sim_s0:,3], 'b-', linewidth=1, label='tau_cmd_knee_d_sim')
    ax.legend(loc='upper right')
    ax.grid()

    i_axis += 1
    ax = axes[i_axis]
    ax.plot(t_sim, traj['a'][sim_s0:,0], 'r-', linewidth=1, label='a0_sim')
    ax.plot(t_sim, traj['a'][sim_s0:,1], 'b-', linewidth=1, label='a1_sim')
    if plot_raw:
        ax.plot(t_sim, traj['a_raw'][sim_s0:,0], 'r-', alpha=0.2, linewidth=1, label='a0_raw_sim')
        ax.plot(t_sim, traj['a_raw'][sim_s0:,1], 'b-', alpha=0.2, linewidth=1, label='a1_raw_sim')
    ax.legend(loc='upper right')
    ax.grid()

    i_axis += 1
    ax = axes[i_axis]
    ax.plot(t_sim, traj['desired_joint_angle'][sim_s0:,0], 'r-', linewidth=1, label='desired_theta_hip_sim')
    ax.plot(t_sim, traj['desired_joint_angle'][sim_s0:,1], 'b-', linewidth=1, label='desired_theta_knee_sim')
    ax.legend(loc='upper right')
    ax.grid()

    i_axis += 1
    ax = axes[i_axis]
    ax.plot(t_sim, traj['desired_foot_offset'][sim_s0:,0], 'r-', linewidth=1, label='desired_foot_offset_x_sim')
    ax.plot(t_sim, traj['desired_foot_offset'][sim_s0:,1], 'b-', linewidth=1, label='desired_foot_offset_y_sim')
    ax.plot(t_sim, traj['desired_foot_offset'][sim_s0:,2], 'g-', linewidth=1, label='desired_foot_offset_z_sim')
    ax.legend(loc='upper right')
    ax.grid()

    i_axis += 1
    ax = axes[i_axis]
    ax.plot(t_sim, traj['foot_offset'][sim_s0:,0], 'r-', linewidth=1, label='foot_offset_x_sim')
    ax.plot(t_sim, traj['foot_offset'][sim_s0:,1], 'b-', linewidth=1, label='foot_offset_y_sim')
    ax.plot(t_sim, traj['foot_offset'][sim_s0:,2], 'g-', linewidth=1, label='foot_offset_z_sim')
    if plot_raw:
        ax.plot(t_sim, traj['foot_offset_raw'][sim_s0:,0], 'r-', alpha=0.2, linewidth=1, label='foot_offset_x_raw_sim')
        ax.plot(t_sim, traj['foot_offset_raw'][sim_s0:,1], 'b-', alpha=0.2, linewidth=1, label='foot_offset_y_raw_sim')
        ax.plot(t_sim, traj['foot_offset_raw'][sim_s0:,2], 'g-', alpha=0.2, linewidth=1, label='foot_offset_z_raw_sim')
    if exp_data is not None:
        ax.plot(t_exp, exp_data['p'][s0:s1,0], 'r--', linewidth=1, label='foot_offset_x_exp')
        ax.plot(t_exp, exp_data['p'][s0:s1,1], 'b--', linewidth=1, label='foot_offset_y_exp')
        ax.plot(t_exp, exp_data['p'][s0:s1,2], 'g--', linewidth=1, label='foot_offset_z_exp')
    ax.legend(loc='upper right')
    ax.grid()

    i_axis += 1
    ax = axes[i_axis]
    ax.plot(t_sim, traj['foot_offset_vel'][sim_s0:,0], 'r-', linewidth=1, label='foot_offset_vel_x_sim')
    ax.plot(t_sim, traj['foot_offset_vel'][sim_s0:,1], 'b-', linewidth=1, label='foot_offset_vel_y_sim')
    ax.plot(t_sim, traj['foot_offset_vel'][sim_s0:,2], 'g-', linewidth=1, label='foot_offset_vel_z_sim')
    if plot_raw:
        ax.plot(t_sim, traj['foot_offset_vel_raw'][sim_s0:,0], 'r-', alpha=0.2, linewidth=1, label='foot_offset_vel_x_raw_sim')
        ax.plot(t_sim, traj['foot_offset_vel_raw'][sim_s0:,1], 'b-', alpha=0.2, linewidth=1, label='foot_offset_vel_y_raw_sim')
        ax.plot(t_sim, traj['foot_offset_vel_raw'][sim_s0:,2], 'g-', alpha=0.2, linewidth=1, label='foot_offset_vel_z_raw_sim')
    if exp_data is not None:
        ax.plot(t_exp, exp_data['v'][s0:s1,0], 'r--', linewidth=1, label='foot_offset_vel_x_exp')
        ax.plot(t_exp, exp_data['v'][s0:s1,1], 'b--', linewidth=1, label='foot_offset_vel_y_exp')
        ax.plot(t_exp, -exp_data['v'][s0:s1,2], 'g--', linewidth=1, label='foot_offset_vel_z_exp')
    ax.legend(loc='upper right')
    ax.grid()

    if exp_data is not None:
        i_axis += 1
        ax = axes[i_axis]
        ax.plot(t_sim, traj['foot_offset_vel'][sim_s0:,0], 'r-', linewidth=1, label='foot_offset_vel_x_sim')
        ax.plot(t_sim, traj['foot_offset_vel'][sim_s0:,1], 'b-', linewidth=1, label='foot_offset_vel_y_sim')
        ax.plot(t_sim, traj['foot_offset_vel'][sim_s0:,2], 'g-', linewidth=1, label='foot_offset_vel_z_sim')
        if plot_raw:
            ax.plot(t_sim, traj['foot_offset_vel_raw'][sim_s0:,0], 'r-', alpha=0.2, linewidth=1, label='foot_offset_vel_x_raw_sim')
            ax.plot(t_sim, traj['foot_offset_vel_raw'][sim_s0:,1], 'b-', alpha=0.2, linewidth=1, label='foot_offset_vel_y_raw_sim')
            ax.plot(t_sim, traj['foot_offset_vel_raw'][sim_s0:,2], 'g-', alpha=0.2, linewidth=1, label='foot_offset_vel_z_raw_sim')
        ax.legend(loc='upper right')
        ax.grid()

    if exp_data is not None:
        i_axis += 1
        ax = axes[i_axis]
        ax.plot(t_exp, exp_data['v'][s0:s1,0], 'r--', linewidth=1, label='foot_offset_vel_x_exp')
        ax.plot(t_exp, exp_data['v'][s0:s1,1], 'b--', linewidth=1, label='foot_offset_vel_y_exp')
        ax.plot(t_exp, -exp_data['v'][s0:s1,2], 'g--', linewidth=1, label='foot_offset_vel_z_exp')
        ax.legend(loc='upper right')
        ax.grid()

    if plot_true:
        i_axis += 1
        ax = axes[i_axis]
        ax.plot(t_sim, traj['foot_offset'][sim_s0:,0], 'r-', linewidth=1, label='foot_offset_x_sim')
        ax.plot(t_sim, traj['foot_offset'][sim_s0:,1], 'b-', linewidth=1, label='foot_offset_y_sim')
        ax.plot(t_sim, traj['foot_offset'][sim_s0:,2], 'g-', linewidth=1, label='foot_offset_z_sim')
        ax.plot(t_sim, traj['foot_offset_true'][sim_s0:,0], 'r--', linewidth=1, label='foot_offset_true_x_sim')
        ax.plot(t_sim, traj['foot_offset_true'][sim_s0:,1], 'b--', linewidth=1, label='foot_offset_true_y_sim')
        ax.plot(t_sim, traj['foot_offset_true'][sim_s0:,2], 'g--', linewidth=1, label='foot_offset_true_z_sim')
        ax.legend(loc='upper right')
        ax.grid()

    if plot_true:
        i_axis += 1
        ax = axes[i_axis]
        ax.plot(t_sim, traj['foot_offset_vel'][sim_s0:,0], 'r-', linewidth=1, label='foot_offset_vel_x_sim')
        ax.plot(t_sim, traj['foot_offset_vel'][sim_s0:,1], 'b-', linewidth=1, label='foot_offset_vel_y_sim')
        ax.plot(t_sim, traj['foot_offset_vel'][sim_s0:,2], 'g-', linewidth=1, label='foot_offset_vel_z_sim')
        ax.plot(t_sim, traj['foot_offset_vel_true'][sim_s0:,0], 'r--', linewidth=1, label='foot_offset_vel_true_x_sim')
        ax.plot(t_sim, traj['foot_offset_vel_true'][sim_s0:,1], 'b--', linewidth=1, label='foot_offset_vel_true_y_sim')
        ax.plot(t_sim, traj['foot_offset_vel_true'][sim_s0:,2], 'g--', linewidth=1, label='foot_offset_vel_true_z_sim')
        ax.legend(loc='upper right')
        ax.grid()

    i_axis += 1
    ax = axes[i_axis]
    ax.plot(t_sim, traj['hip_pos'][sim_s0:,2], 'r-', linewidth=1, label='hip_pos_z_sim')
    ax.plot(t_sim, traj['knee_pos'][sim_s0:,2], 'b-', linewidth=1, label='knee_pos_z_sim')
    ax.plot(t_sim, traj['foot_pos'][sim_s0:,2], 'g-', linewidth=1, label='foot_pos_z_sim')
    ax.plot(t_sim, traj['leg_pos'][sim_s0:,2], 'k-', linewidth=1, label='leg_pos_z_sim')
    ax.legend(loc='upper right')
    ax.grid()

    i_axis += 1
    ax = axes[i_axis]
    ax.plot(t_sim, traj['hip_pos'][sim_s0:,0], 'r-', linewidth=1, label='hip_pos_x_sim')
    ax.plot(t_sim, traj['knee_pos'][sim_s0:,0], 'b-', linewidth=1, label='knee_pos_x_sim')
    ax.plot(t_sim, traj['foot_pos'][sim_s0:,0], 'g-', linewidth=1, label='foot_pos_x_sim')
    ax.plot(t_sim, traj['leg_pos'][sim_s0:,0], 'k-', linewidth=1, label='leg_pos_x_sim')
    ax.legend(loc='upper right')
    ax.grid()

    i_axis += 1
    ax = axes[i_axis]
    ax.plot(t_sim, traj['hip_vel'][sim_s0:,2], 'r-', linewidth=1, label='hip_vel_z_sim')
    ax.plot(t_sim, traj['knee_vel'][sim_s0:,2], 'b-', linewidth=1, label='knee_vel_z_sim')
    ax.plot(t_sim, traj['foot_vel'][sim_s0:,2], 'g-', linewidth=1, label='foot_vel_z_sim')
    ax.plot(t_sim, traj['leg_vel'][sim_s0:,2], 'k-', linewidth=1, label='leg_vel_z_sim')
    ax.legend(loc='upper right')
    ax.grid()

    i_axis += 1
    ax = axes[i_axis]
    ax.plot(t_sim, traj['hip_vel'][sim_s0:,0], 'r-', linewidth=1, label='hip_vel_x_sim')
    ax.plot(t_sim, traj['knee_vel'][sim_s0:,0], 'b-', linewidth=1, label='knee_vel_x_sim')
    ax.plot(t_sim, traj['foot_vel'][sim_s0:,0], 'g-', linewidth=1, label='foot_vel_x_sim')
    ax.plot(t_sim, traj['leg_vel'][sim_s0:,0], 'k-', linewidth=1, label='leg_vel_x_sim')
    ax.legend(loc='upper right')
    ax.grid()

    i_axis += 1
    ax = axes[i_axis]
    ax.plot(t_sim, traj['work'][sim_s0:], 'k-', linewidth=1, label='work_sim')
    ax.plot(t_sim, traj['work_hip'][sim_s0:], 'r-', linewidth=1, label='work_hip_sim')
    ax.plot(t_sim, traj['work_knee'][sim_s0:], 'b-', linewidth=1, label='work_knee_sim')
    if exp_data is not None:
        ax.plot(t_exp, exp_data['work'][s0:s1], 'k--', linewidth=1, label='work_exp')
        ax.plot(t_exp, exp_data['work_hip'][s0:s1], 'r--', linewidth=1, label='work_hip_exp')
        ax.plot(t_exp, exp_data['work_knee'][s0:s1], 'b--', linewidth=1, label='work_knee_exp')
    ax.legend(loc='upper right')
    ax.grid()

    i_axis += 1
    ax = axes[i_axis]
    ax.plot(t_sim, traj['E'][sim_s0:], 'k-', linewidth=1, label='energy_sim')
    ax.plot(t_sim, traj['KE'][sim_s0:], 'r-', linewidth=1, label='kinetic_sim')
    ax.plot(t_sim, traj['PE'][sim_s0:], 'b-', linewidth=1, label='potential_sim')
    if exp_data is not None:
        ax.plot(t_exp_eng, exp_data['E'][exp_eng_s0:exp_eng_s1], 'k--', linewidth=1, label='energy_exp')
        ax.plot(t_exp_eng, exp_data['KE'][exp_eng_s0:exp_eng_s1], 'r--', linewidth=1, label='kinetic_exp')
        ax.plot(t_exp_eng, exp_data['PE'][exp_eng_s0:exp_eng_s1], 'b--', linewidth=1, label='potential_exp')
    ax.legend(loc='upper right')
    ax.grid()

    i_axis += 1
    ax = axes[i_axis]
    ax.plot(t_sim, traj['energy_loss'][sim_s0:], 'k-', linewidth=1, label='energy_loss_sim')
    if exp_data is not None:
        ax.plot(t_exp_eng, exp_data['energy_loss'][exp_eng_s0:exp_eng_s1], 'k--', linewidth=1, label='energy_loss_exp')
    ax.legend(loc='upper right')
    ax.grid()

    i_axis += 1
    ax = axes[i_axis]
    ax.plot(t_sim, traj['foot_in_contact'][sim_s0:], 'g-', linewidth=1, label='foot_in_contact_sim')
    ax.plot(t_sim, traj['is_jumping'][sim_s0:], 'k-', linewidth=1, label='is_jumping_sim')
    if exp_data is not None and 'is_jumping' in exp_data:
        ax.plot(t_exp, exp_data['is_jumping'][s0:s1], 'k--', linewidth=1, label='is_jumping_exp')
    ax.legend(loc='upper right')
    ax.grid()

    ax.set_xlabel('time (s)')

    fig.set_tight_layout(dict(h_pad=0.1))
    ret = []
    if return_fig:
        ret.append(fig)
    if return_traj:
        ret.append(traj)
    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)

# FIXME: broken
def simulate(sim_env, policy):
    viewer.launch(sim_env, policy=policy)

def simulate_and_record(sim_interface, policy, fps=50, filename='video'):
    import cv2

    def grabMujocoFrame(env, height=600, width=480, camera_id='fixed-cam'):
        rgbArr = env.physics.render(height, width, camera_id=camera_id)
        return cv2.cvtColor(rgbArr, cv2.COLOR_BGR2RGB)

    frame = grabMujocoFrame(sim_interface.sim_env)
    height, width, layers = frame.shape
    video = cv2.VideoWriter(filename + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    s = sim_interface.reset()
    done = False
    while not done:
        a = policy(s)
        s, r, done = sim_interface.step(a)
        frame = grabMujocoFrame(sim_interface.sim_env)
        video.write(frame)
    video.release()

def write_snapshot_info(snapshot_header, snapshot_info, logdir_base=None):

    def id2num(id):
        """Takes a job id like '23235.domain.net' and returns '23235', or None if the id doesn't start with a number."""
        if id is None:
            return None
        num = None
        m = re.match('^([0-9]+)', id)
        if m:
            num = m.group(1)
        return num

    job_info = {
        'slurm_jobname': os.environ.get('SLURM_JOB_NAME', None),
        'slurm_o_workdir': os.environ.get('SLURM_SUBMIT_DIR', None),
        'slurm_jobnum': id2num(os.environ.get('SLURM_JOB_ID', None)),
        'slurm_o_host': os.environ.get('SLURM_SUBMIT_HOST', None),
        'slurm_jobid': os.environ.get('SLURM_JOB_ID', None),
    }

    snapshot_info_file = 'job'
    if job_info['slurm_jobname'] is not None:
        snapshot_info_file += '-' + job_info['slurm_jobname']
    snapshot_info_file += '.csv'

    if logdir_base is None:
        logdir_base = 'logdir'
    logdir = os.path.join(logdir_base, snapshot_info_file)

    line_to_write = snapshot_info
    csv_header = snapshot_header
    if job_info['slurm_jobnum'] is not None:
        csv_header = 'JobID,' + csv_header
        line_to_write = job_info['slurm_jobnum'] + ',' + line_to_write
    csv_header += '\n'
    line_to_write += '\n'

    if os.path.isfile(logdir):
        with open(logdir, 'a') as f:
            f.write(line_to_write)
    else:
        with open(logdir, 'w') as f:
            f.write(csv_header)
            f.write(line_to_write)

if __name__ == '__main__':
    main()
