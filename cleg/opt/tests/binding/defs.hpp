/////////////////////////////////////////////////////
////////////////// Debug Options ////////////////////
/////////////////////////////////////////////////////

#define Debug_main
#undef Debug_step_inner
#undef Debug_step_outer
#undef Debug_reward

/////////////////////////////////////////////////////
/////////////// Argument Definitions ////////////////
/////////////////////////////////////////////////////

#define True true
#define False false

#define compliant 0
#define noncompliant 1

#define torque 0
#define jointspace_pd 1
#define workspace_pd 2

#define file_path_type 0
#define content_type 1

#define tanh_activation 0
#define relu_activation 1

#define PI 3.14159265358979323846

/////////////////////////////////////////////////////
/////////////// SimIF Options Defs //////////////////
/////////////////////////////////////////////////////

#ifndef xml_type
  #define xml_type file_path_type
#endif

#ifndef xml_file
  #define xml_file "./leg.xml"
#endif

#define inner_loop_rate 4000
#define outer_loop_rate 4000
#define ground_contact_type compliant
#define physics_timestep 0.00025
#define motor_root_real -1606.059046034205
#define motor_root_imag 4190.778464532961
#define torque_delay 0.001
#define observation_delay 0.001
#define maxabs_tau 10
#define output_torque_scale 0.8
#define time_before_reset 2.0
#define hip_kP 1.25
#define hip_kD 0.05
#define knee_kP 1.25
#define knee_kD 0.05

#define maxabs_omega_for_tau 21.66
#define tau_omega_slope -0.397
#define action_type torque
#define use_motor_model True
#define do_jump False
#define simplified_reward True

//disabled
#define do_obs_noise False
#define contact_obs False
#define jumping_obs False
#define extra_obs_dim False
#define filter_state False
#define filter_action False
#define action_is_delta False
//disabled reward stuff
#define add_torque_penalty_on_air False
#define add_omega_smoothness_reward False

/////////////////////////////////////////////////////
////////////// MLP Module Definitions ///////////////
/////////////////////////////////////////////////////

#define h1 64  // Hidden Units in the MLP's 1st Layer
#define h2 64  // Hidden Units in the MLP's 2nd Layer
#define activation tanh_activation

/////////////////////////////////////////////////////
///////////  Checking Multi-choice options //////////
/////////////////////////////////////////////////////

#if !defined(ground_contact_type)                                || \
    !((ground_contact_type == compliant)                         || \
      (ground_contact_type == noncompliant))
  #error "Undefined ground_contact_type."
#endif

#if !defined(action_type) || !((action_type == torque)           || \
                               (action_type == jointspace_pd)    || \
                               (action_type == workspace_pd))
  #error "Undefined action_type."
#endif

#if !defined(do_obs_noise) || !((do_obs_noise == True)           || \
                                (do_obs_noise == False))
  #error "Undefined do_obs_noise."
#endif

#if !defined(do_jump) || !((do_jump == True)                     || \
                           (do_jump == False))
  #error "Undefined do_jump."
#endif

#if !defined(use_motor_model) || !((use_motor_model == True)     || \
                                   (use_motor_model == False))
  #error "Undefined use_motor_model."
#endif

#if !defined(contact_obs) || !((contact_obs == True)             || \
                               (contact_obs == False))
  #error "Undefined contact_obs."
#endif

#if !defined(jumping_obs) || !((jumping_obs == True)             || \
                               (jumping_obs == False))
  #error "Undefined jumping_obs."
#endif

#if !defined(extra_obs_dim) || !((extra_obs_dim == True)         || \
                                 (extra_obs_dim == False))
  #error "Undefined extra_obs_dim."
#endif

#if !defined(filter_state) || !((filter_state == True)           || \
                                (filter_state == False))
  #error "Undefined filter_state."
#endif

#if !defined(filter_action) || !((filter_action == True)         || \
                                 (filter_action == False))
  #error "Undefined filter_action."
#endif

#if !defined(action_is_delta) || !((action_is_delta == True)     || \
                                   (action_is_delta == False))
  #error "Undefined action_is_delta."
#endif

#if !defined(simplified_reward) || !((simplified_reward== True)  || \
                                     (simplified_reward== False))
  #error "Undefined simplified_reward."
#endif


#if !defined(add_torque_penalty_on_air)                          || \
    !((add_torque_penalty_on_air == True)                        || \
      (add_torque_penalty_on_air == False))
  #error "Undefined add_torque_penalty_on_air."
#endif

#if !defined(add_omega_smoothness_reward)                        || \
    !((add_omega_smoothness_reward == True)                      || \
      (add_omega_smoothness_reward == False))
  #error "Undefined add_omega_smoothness_reward."
#endif

#if !defined(activation) || !((activation == tanh_activation)    || \
                              (activation == relu_activation))
  #error "Undefined activation."
#endif

/////////////////////////////////////////////////////
/////////  Disabled/not-implemented options /////////
/////////////////////////////////////////////////////

#ifdef obs_history_taps
  #error "obs_history_taps is not implemented in this C++ wrapper."
#endif

#ifdef obs_history_len
  #error "obs_history_len is not implemented in this C++ wrapper."
#else
  #define obs_history_len_int 1
#endif

#if contact_obs == True
  #error "contact_obs is not implemented in this C++ wrapper."
#else
  #define contact_obs_int 0
#endif

#if jumping_obs == True
  #error "jumping_obs is not implemented in this C++ wrapper."
#else
  #define jumping_obs_int 0
#endif

#if extra_obs_dim == True
  #error "extra_obs_dim is not implemented in this C++ wrapper."
#else
  #define extra_obs_dim_int 0
#endif

#if filter_state == True
  #error "filter_state is not implemented in this C++ wrapper."
#endif

#if filter_action == True
  #error "filter_action is not implemented in this C++ wrapper."
#endif

#if do_obs_noise == True
  #error "do_obs_noise is under construction in this C++ wrapper."
#endif

#if do_jump == True
  #error "do_jump == True is under construction in this C++ wrapper."
#endif

#if action_type == workspace_pd
  #error "workspace_pd is not implemented in this C++ wrapper."
#endif

#if action_is_delta == True
  #error "action_is_delta is not implemented in this C++ wrapper."
#endif

#if simplified_reward == False
  #error "only simplified_reward is implemented in this C++ wrapper."
#else
  #define RewardGiver SimplifiedRewardGiver
#endif

#if add_torque_penalty_on_air == True
  #error "add_torque_penalty_on_air is not implemented in this C++ wrapper."
#endif

#if add_omega_smoothness_reward == True
  #error "add_omega_smoothness_reward is not implemented in this C++ wrapper."
#endif

// Functional definitions
// TODO: Compute these numbers in python and pass them as options.
//       We absolutely don't want these numbers to be computed
//       over and over again or consume unnecessary memory!
#define inner_step_time 0.00025
#define inner_loops_per_outer_loop 1
#define physics_loops_per_inner_loop 1
#define obs_dim 4
#define act_dim 2
#define torque_delay_steps 4
#define observation_delay_steps 4
#define trq_dly_buflen 5
#define obs_dly_buflen 5
#define done_inner_steps 8000

#define fc1_size  (obs_dim * h1)
#define fc2_size  (h1      * h2)
#define fc3_size  (h2 * act_dim)
#define h1_div_4  (h1/4)
#define h2_div_4  (h2/4)

/////////////////////////////////////////////////////
///////// C++ Wrapper Specific Definitions //////////
/////////////////////////////////////////////////////

#define mjstep1_after_mjstep     0
#define separate_mjstep1_mjstep2 1
#define delay_valid_obs          2

#define mjstep_order separate_mjstep1_mjstep2

#if mjstep_order == mjstep1_after_mjstep
  #define rew_dly_buflen (obs_dly_buflen)
#elif mjstep_order == separate_mjstep1_mjstep2
  #if observation_delay_steps < 1
    #error "You cannot logically use (mjstep_order = separate_mjstep1_mjstep2) "
           "when observation_delay_steps is less than 1. You may want to use the "
           "inefficient mjstep1_after_mjstep mode if you insist!"
  #endif
  #define rew_dly_buflen (obs_dly_buflen - 1)
#elif mjstep_order == delay_valid_obs
  #if observation_delay_steps < 1
    #error "You cannot logically use (mjstep_order = delay_valid_obs) when"
           "observation_delay_steps is less than 1. You may want to use the "
           "inefficient mjstep1_after_mjstep mode if you insist!"
  #endif
  #define rew_dly_buflen (obs_dly_buflen - 1)
#else
  #error "Unknown mjstep_order"
#endif

/////////////////////////////////////////////////////
////////////// Main Program Definitions /////////////
/////////////////////////////////////////////////////

#define SimIF_CPP 0
#define MlpIF_CPP 1
#define Rollout_CPP 2

#ifndef __MAINPROG__
  #define __MAINPROG__ SimIF_CPP
#endif

#if !defined(__MAINPROG__) || !((__MAINPROG__ == SimIF_CPP)                     || \
                                (__MAINPROG__ == MlpIF_CPP)                     || \
                                (__MAINPROG__ == Rollout_CPP))
  #error "Undefined __MAINPROG__."
#endif
