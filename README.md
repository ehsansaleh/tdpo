## The Stochastic Drop Test (Successfully Tested on Hardware)

This branch builds on the `07_drop_dense_rew` branch with `torque_smoothness_coefficient=2000` (i.e., 5x the usual value).

Mainly, this was an effort to produce smoother agents to be tested on hardware. The hardware results were successful, and the agent produced non-oscillatory behavior. A direct `torque` control agent using 4 kHz inner and outer loop rates with wide initialization state range and observation noise was trained in the `agents` directory.

Here are the relevant interface options of interest:
```
"inner_loop_rate": 4000,
"outer_loop_rate": 4000,
"action_type": "torque",
"stand_reward": "dense",
"torque_smoothness_coefficient": 2000,
"do_obs_noise": true,
"theta_hip_init": [-3.14, 0.52],
"theta_knee_init": [-2.70, -0.61],
"pos_slider_init": [0.4, 0.8],
"omega_hip_init": [0, 0],
"omega_knee_init": [0, 0],
"vel_slider_init": [0, 0],
"ground_contact_type": "compliant",
"time_before_reset": 2,
"use_motor_model": true,
"torque_delay": 0.001,
"observation_delay": 0.001,
"output_torque_scale": 1,
```
