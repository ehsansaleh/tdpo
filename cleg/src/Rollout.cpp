#include "mujoco.h"
#include "iostream"
#include <iomanip>    // std::setprecision, std::setw
#include <chrono>
#include <cstring>    // strcpy

#include "defs.hpp"
#include "Rollout.hpp"

#if xml_type == content_type
  #include "mj_xml.hpp"    // xml_bytes, xml_content
#endif


inline void addTo_act(double* src, double* dst){
  #if act_dim == 2
    dst[0] += src[0];
    dst[1] += src[1];
  #else
    mju_addTo(dst, src, act_dim);
  #endif
}

inline void copy_act(double* src, double* dst){
  #if act_dim == 2
    dst[0] = src[0];
    dst[1] = src[1];
  #else
    mju_copy(dst, src, act_dim);
  #endif
}

inline void copy_obs(double* src, double* dst){
  #if obs_dim == 4
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];
  #else
    mju_copy(dst, src, obs_dim);
  #endif
}

Rollout::Rollout(void){
  // No need to do anything!
}

void Rollout::greedy_lite(int traj_num, int n_steps, double gamma,
                          double* eta_greedy, double* return_greedy,
                          int* done_steps){
  for (traj_idx=0; traj_idx < traj_num; traj_idx++){
    // Resetting the Sim Interface
    observation = simiface.reset(theta_hip_inits[traj_idx],
      theta_knee_inits[traj_idx], omega_hip_inits[traj_idx],
      omega_knee_inits[traj_idx], pos_slider_inits[traj_idx],
      vel_slider_inits[traj_idx], jumping_time_inits[traj_idx],
      noise_index_inits[traj_idx]);

    gamma_pow = 1;
    eta_greedy[traj_idx] = 0;
    return_greedy[traj_idx] = 0;
    for (step = 0; step < n_steps; step++){
      mlp_action = net.forward(observation);
      simiface.step(mlp_action, &observation, &reward, &done);
      eta_greedy[traj_idx] += gamma_pow * reward;
      return_greedy[traj_idx] += reward;
      gamma_pow *= gamma;
      if (done) break;
    }
    done_steps[traj_idx] = step + (int) done;
    // (int) done is added for the edge case of breaking out!
  }
}

void Rollout::vine_lite(int traj_num, int n_steps, double gamma,
                        int expl_steps, int* reset_times, double* expl_noise,
                        double* obs_greedy, double* action_greedy, double* action_vine,
                        double* Q_greedy, double* eta_greedy, double* return_greedy,
                        double* Q_vine, double* eta_vine, double* return_vine,
                        int* done_steps, int* done_steps_vine){
  int active_traj_len;
  for (traj_idx=0; traj_idx < traj_num; traj_idx++){
    for (bool do_exploration : { false, true }){
      // Resetting the Sim Interface
      done = false;
      observation = simiface.reset(theta_hip_inits[traj_idx],
        theta_knee_inits[traj_idx], omega_hip_inits[traj_idx],
        omega_knee_inits[traj_idx], pos_slider_inits[traj_idx],
        vel_slider_inits[traj_idx], jumping_time_inits[traj_idx],
        noise_index_inits[traj_idx]);

      gamma_pow = 1;
      gamma_pow_q = 1;
      eta = (do_exploration) ? eta_vine : eta_greedy;
      return_ = (do_exploration) ? return_vine : return_greedy;
      Q = (do_exploration) ? Q_vine : Q_greedy;
      eta[traj_idx] = 0;
      return_[traj_idx] = 0;
      Q[traj_idx] = 0;

      st_expl = reset_times[traj_idx];
      end_expl = st_expl + expl_steps;

      gen_idx = traj_idx * expl_steps;
      obs_g_idx = gen_idx * obs_dim;
      act_idx  = gen_idx * act_dim;
      active_traj_len = 0;

      for (step = 0; step < n_steps; step++){
        mlp_action = net.forward(observation);
        if ((step >= st_expl) && (step < end_expl)){
          if (do_exploration){
            addTo_act(expl_noise + act_idx, mlp_action);
            copy_act(mlp_action, action_vine + act_idx);
          } else {
            copy_obs(observation, obs_greedy + obs_g_idx);
            copy_act(mlp_action, action_greedy + act_idx);
            obs_g_idx += obs_dim;
          }
          act_idx += act_dim;
        }
        if (!done) {
          simiface.step(mlp_action, &observation, &reward, &done);
          active_traj_len = step + 1;
        } else {
          // Since the env is already done, let's pretend that the
          // observation is frozen, and we're taking the same action.
          reward = 0;
        }
        eta[traj_idx] += gamma_pow * reward;
        return_[traj_idx] += reward;
        gamma_pow *= gamma;
        if (step >= st_expl) {
          Q[traj_idx] += gamma_pow_q * reward;
          gamma_pow_q *= gamma;
        }
      }
      if (!do_exploration) {
        done_steps[traj_idx] = active_traj_len;
      } else {
        done_steps_vine[traj_idx] = active_traj_len;
      }
    }
  }
}

void Rollout::stochastic(int traj_num, int n_steps, double* expl_noise,
                         double* obs, double* action, double* rewards,
                         int* done_steps){
  int act_idx=0;
  int obs_idx=0;
  int gen_idx=0;
  double* rew_ptr;

  for (traj_idx=0; traj_idx < traj_num; traj_idx++){
      // Resetting the Sim Interface
      observation = simiface.reset(theta_hip_inits[traj_idx],
        theta_knee_inits[traj_idx], omega_hip_inits[traj_idx],
        omega_knee_inits[traj_idx], pos_slider_inits[traj_idx],
        vel_slider_inits[traj_idx], jumping_time_inits[traj_idx],
        noise_index_inits[traj_idx]);

      gen_idx = (traj_idx * n_steps);
      rew_ptr = rewards + gen_idx;
      for (step = 0; step < n_steps; step++){
        act_idx = gen_idx * act_dim;
        obs_idx = gen_idx * obs_dim;
        gen_idx++;

        mlp_action = action + act_idx;
        net.forward(observation, mlp_action);
        addTo_act(expl_noise + act_idx, mlp_action);
        copy_obs(observation, obs + obs_idx);

        simiface.step(mlp_action, &observation, rew_ptr, &done);
        rew_ptr++;
        if (done) break;
      }
      done_steps[traj_idx] = step + (int) done;
      // (int) done is added for the edge case of breaking out!
  }
}

void Rollout::reset(int traj_idx){
  observation = simiface.reset(theta_hip_inits[traj_idx],
    theta_knee_inits[traj_idx], omega_hip_inits[traj_idx],
    omega_knee_inits[traj_idx], pos_slider_inits[traj_idx],
    vel_slider_inits[traj_idx], jumping_time_inits[traj_idx],
    noise_index_inits[traj_idx]);
}

#include <chrono>

void Rollout::partial_stochastic(int n_steps, double* expl_noise, double* obs,
                                 double* action, double* rewards, bool* dones){
  for (step = 0; step < n_steps; step++){
    mlp_action = action + step * act_dim;
    net.forward(observation, mlp_action);
    addTo_act(expl_noise + step * act_dim, mlp_action);
    copy_obs(observation, obs + step * obs_dim);
    simiface.step(mlp_action, &observation, rewards + step, &done); // 0.8998 sec for 14 trajs
    dones[step] = done;
    if (done) break;
  }
}

// These will be used by python's ctypes library for binding
extern "C"
{
  Rollout* rollout_new() {return new Rollout();}

  void rollout_set_simif_inits(Rollout* rollout, double* theta_hip, double* theta_knee,
    double* omega_hip, double* omega_knee, double* pos_slider, double* vel_slider,
    double* jumping_time, int* noise_index){
    rollout->set_simif_inits(theta_hip, theta_knee,
      omega_hip, omega_knee, pos_slider, vel_slider,
      jumping_time, noise_index);
  }

  void rollout_set_mlp_weights(Rollout* rollout, double* fc1, double* fc2, double* fc3,
    double* fc1_bias, double* fc2_bias, double* fc3_bias){
    rollout->set_mlp_weights(fc1, fc2, fc3, fc1_bias, fc2_bias, fc3_bias);
  }

  void rollout_greedy_lite(Rollout* rollout, int traj_num, int n_steps, double gamma,
    double* eta_greedy, double* return_greedy,
    int* done_steps) {
    rollout->greedy_lite(traj_num, n_steps, gamma,
      eta_greedy, return_greedy,
      done_steps);
  }

  void rollout_vine_lite(Rollout* rollout, int traj_num, int n_steps, double gamma, int expl_steps,
    int* reset_times, double* expl_noise, double* obs_greedy, double* action_greedy, double* action_vine,
    double* Q_greedy, double* eta_greedy, double* return_greedy, double* Q_vine, double* eta_vine,
    double* return_vine, int* done_steps, int* done_steps_vine) {
    rollout->vine_lite(traj_num, n_steps, gamma, expl_steps, reset_times, expl_noise,
        obs_greedy, action_greedy, action_vine, Q_greedy, eta_greedy, return_greedy,
        Q_vine, eta_vine, return_vine, done_steps, done_steps_vine);
  }

  void rollout_stochastic(Rollout* rollout, int traj_num, int n_steps,
    double* expl_noise, double* obs, double* action, double* rewards,
    int* done_steps){
    rollout->stochastic(traj_num, n_steps, expl_noise, obs,
      action, rewards, done_steps);
  }

  void rollout_infer_mlp(Rollout* rollout, int input_num,
    double* mlp_input, double* mlp_output) {
    rollout->infer_mlp(input_num, mlp_input, mlp_output);
  }

  void rollout_partial_stochastic(Rollout* rollout, int n_steps, double* expl_noise,
    double* obs, double* action, double* rewards, bool* dones) {
    rollout->partial_stochastic(n_steps, expl_noise, obs,
                                action, rewards, dones);
  }

  void rollout_reset(Rollout* rollout, int traj_idx) {
    rollout->reset(traj_idx);
  }


  void rollout_get_build_options(Rollout* rollout, char* keys, double* vals, char* xml_var,
    int keys_len, int vals_len, int xml_var_len) {
    get_build_options(keys, vals, keys_len, vals_len);
    #if xml_type == file_path_type
      strcpy(xml_var, xml_path);
    #elif xml_type == content_type
      for (int i=0; i<(xml_bytes-5); i++) {
        xml_var[i] = xml_content[i];
        if (i >= xml_var_len) {
          throw std::runtime_error("xml_var_len is too small");
          break;
        }
      }
    #endif
  }
}

void write_opt(const std::string key_str, double val, char** key_write_ptr,
               double** val_write_ptr, char* key_write_ptr_max,
               double* val_write_ptr_max) {
  int n = key_str.length() + 1;
  if ((*key_write_ptr + n) >= key_write_ptr_max)
    throw std::runtime_error("the destination char array is too small");
  if ((*val_write_ptr + 1) >= val_write_ptr_max)
    throw std::runtime_error("the destination char array is too small");
  strcpy(*key_write_ptr, key_str.c_str());
  **val_write_ptr = val;
  *key_write_ptr += n;
  *val_write_ptr = *val_write_ptr + 1;
}

void get_build_options(char* keys, double* vals, int keys_len, int vals_len) {
  char* key_p;
  double* val_p;
  char* a = keys + keys_len;
  double* b = vals + vals_len;
  key_p = keys;
  val_p = vals;
  // enumerations and constants
  write_opt("True", True, &key_p, &val_p, a, b);
  write_opt("False", False, &key_p, &val_p, a, b);
  write_opt("compliant_v1", compliant_v1, &key_p, &val_p, a, b);
  write_opt("compliant_v2", compliant_v2, &key_p, &val_p, a, b);
  write_opt("noncompliant", noncompliant, &key_p, &val_p, a, b);
  write_opt("torque", torque, &key_p, &val_p, a, b);
  write_opt("jointspace_pd", jointspace_pd, &key_p, &val_p, a, b);
  write_opt("workspace_pd", workspace_pd, &key_p, &val_p, a, b);
  write_opt("file_path_type", file_path_type, &key_p, &val_p, a, b);
  write_opt("content_type", content_type, &key_p, &val_p, a, b);
  write_opt("tanh_activation", tanh_activation, &key_p, &val_p, a, b);
  write_opt("relu_activation", relu_activation, &key_p, &val_p, a, b);
  write_opt("dense", dense, &key_p, &val_p, a, b);
  write_opt("sparse", sparse, &key_p, &val_p, a, b);
  write_opt("simplified", simplified, &key_p, &val_p, a, b);
  write_opt("dyno", dyno, &key_p, &val_p, a, b);
  write_opt("naive", naive, &key_p, &val_p, a, b);
  write_opt("legacy", legacy, &key_p, &val_p, a, b);
  write_opt("PI", PI, &key_p, &val_p, a, b);
  // settings
  write_opt("xml_type", xml_type, &key_p, &val_p, a, b);
  write_opt("inner_loop_rate", inner_loop_rate, &key_p, &val_p, a, b);
  write_opt("outer_loop_rate", outer_loop_rate, &key_p, &val_p, a, b);
  write_opt("ground_contact_type", ground_contact_type, &key_p, &val_p, a, b);
  write_opt("do_obs_noise", do_obs_noise, &key_p, &val_p, a, b);
  write_opt("jumping_obs", jumping_obs, &key_p, &val_p, a, b);
  write_opt("physics_timestep", physics_timestep, &key_p, &val_p, a, b);
  write_opt("torque_delay", torque_delay, &key_p, &val_p, a, b);
  write_opt("observation_delay", observation_delay, &key_p, &val_p, a, b);
  write_opt("time_before_reset", time_before_reset, &key_p, &val_p, a, b);
  write_opt("add_act_sm_r", add_act_sm_r, &key_p, &val_p, a, b);
  write_opt("action_type", action_type, &key_p, &val_p, a, b);
  write_opt("use_motor_model", use_motor_model, &key_p, &val_p, a, b);
  write_opt("motor_saturation_model", motor_saturation_model, &key_p, &val_p, a, b);
  write_opt("use_dyno_model", use_dyno_model, &key_p, &val_p, a, b);
  write_opt("use_naive_model", use_naive_model, &key_p, &val_p, a, b);
  write_opt("use_legacy_model", use_legacy_model, &key_p, &val_p, a, b);
  write_opt("do_jump", do_jump, &key_p, &val_p, a, b);
  write_opt("timed_jump", timed_jump, &key_p, &val_p, a, b);
  write_opt("stand_reward", stand_reward, &key_p, &val_p, a, b);
  write_opt("motor_root_real", motor_root_real, &key_p, &val_p, a, b);
  write_opt("motor_root_imag", motor_root_imag, &key_p, &val_p, a, b);
  write_opt("hip_kP", hip_kP, &key_p, &val_p, a, b);
  write_opt("hip_kD", hip_kD, &key_p, &val_p, a, b);
  write_opt("knee_kP", knee_kP, &key_p, &val_p, a, b);
  write_opt("knee_kD", knee_kD, &key_p, &val_p, a, b);
  write_opt("maxabs_tau", maxabs_tau, &key_p, &val_p, a, b);
  write_opt("output_torque_scale", output_torque_scale, &key_p, &val_p, a, b);
  write_opt("check_mj_unstability", check_mj_unstability, &key_p, &val_p, a, b);
  write_opt("maxabs_omega_for_tau", maxabs_omega_for_tau, &key_p, &val_p, a, b);
  write_opt("tau_omega_slope", tau_omega_slope, &key_p, &val_p, a, b);
  write_opt("omega_noise_scale", omega_noise_scale, &key_p, &val_p, a, b);
  write_opt("noise_rows", noise_rows, &key_p, &val_p, a, b);
  write_opt("jump_push_time", jump_push_time, &key_p, &val_p, a, b);
  write_opt("jump_fly_time", jump_fly_time, &key_p, &val_p, a, b);
  write_opt("torque_smoothness_coeff", torque_smoothness_coeff, &key_p, &val_p, a, b);
  write_opt("posture_height", posture_height, &key_p, &val_p, a, b);
  write_opt("max_leg_vel", max_leg_vel, &key_p, &val_p, a, b);
  write_opt("jump_vel_coeff", jump_vel_coeff, &key_p, &val_p, a, b);
  write_opt("max_reach", max_reach, &key_p, &val_p, a, b);
  write_opt("jump_vel_reward", jump_vel_reward, &key_p, &val_p, a, b);
  write_opt("turn_off_constraints", turn_off_constraints, &key_p, &val_p, a, b);
  write_opt("omega_hip_maxabs", omega_hip_maxabs, &key_p, &val_p, a, b);
  write_opt("omega_knee_maxabs", omega_knee_maxabs, &key_p, &val_p, a, b);
  write_opt("constraint_penalty", constraint_penalty, &key_p, &val_p, a, b);
  write_opt("theta_hip_bounds_low", theta_hip_bounds_low, &key_p, &val_p, a, b);
  write_opt("theta_hip_bounds_high", theta_hip_bounds_high, &key_p, &val_p, a, b);
  write_opt("theta_knee_bounds_low", theta_knee_bounds_low, &key_p, &val_p, a, b);
  write_opt("theta_knee_bounds_high", theta_knee_bounds_high, &key_p, &val_p, a, b);
  write_opt("hip_minimum_z", hip_minimum_z, &key_p, &val_p, a, b);
  write_opt("knee_minimum_z", knee_minimum_z, &key_p, &val_p, a, b);
  write_opt("rpm_per_omega", rpm_per_omega, &key_p, &val_p, a, b);
  write_opt("fcn_k50_co_0", fcn_k50_co_0, &key_p, &val_p, a, b);
  write_opt("fcn_k50_co_1", fcn_k50_co_1, &key_p, &val_p, a, b);
  write_opt("fcn_k50_co_2", fcn_k50_co_2, &key_p, &val_p, a, b);
  write_opt("max_cmd_dyno", max_cmd_dyno, &key_p, &val_p, a, b);
  write_opt("rpm_for_max_cmd_dyno", rpm_for_max_cmd_dyno, &key_p, &val_p, a, b);
  write_opt("cmd_per_rpm_slope_dyno", cmd_per_rpm_slope_dyno, &key_p, &val_p, a, b);
  write_opt("model_coeff_dyno_0", model_coeff_dyno_0, &key_p, &val_p, a, b);
  write_opt("model_coeff_dyno_1", model_coeff_dyno_1, &key_p, &val_p, a, b);
  write_opt("model_coeff_dyno_2", model_coeff_dyno_2, &key_p, &val_p, a, b);
  write_opt("model_coeff_dyno_3", model_coeff_dyno_3, &key_p, &val_p, a, b);
  write_opt("model_coeff_dyno_4", model_coeff_dyno_4, &key_p, &val_p, a, b);
  write_opt("model_coeff_dyno_5", model_coeff_dyno_5, &key_p, &val_p, a, b);
  write_opt("contact_obs", contact_obs, &key_p, &val_p, a, b);
  write_opt("extra_obs_dim", extra_obs_dim, &key_p, &val_p, a, b);
  write_opt("filter_state", filter_state, &key_p, &val_p, a, b);
  write_opt("filter_action", filter_action, &key_p, &val_p, a, b);
  write_opt("action_is_delta", action_is_delta, &key_p, &val_p, a, b);
  write_opt("add_torque_penalty_on_air", add_torque_penalty_on_air, &key_p, &val_p, a, b);
  write_opt("add_omega_smoothness_reward", add_omega_smoothness_reward, &key_p, &val_p, a, b);
  write_opt("h1", h1, &key_p, &val_p, a, b);
  write_opt("h2", h2, &key_p, &val_p, a, b);
  write_opt("activation", activation, &key_p, &val_p, a, b);
  write_opt("obs_history_len_int", obs_history_len_int, &key_p, &val_p, a, b);
  write_opt("contact_obs_int", contact_obs_int, &key_p, &val_p, a, b);
  write_opt("jumping_obs_int", jumping_obs_int, &key_p, &val_p, a, b);
  write_opt("extra_obs_dim_int", extra_obs_dim_int, &key_p, &val_p, a, b);
  write_opt("inner_step_time", inner_step_time, &key_p, &val_p, a, b);
  write_opt("inner_loops_per_outer_loop", inner_loops_per_outer_loop, &key_p, &val_p, a, b);
  write_opt("physics_loops_per_inner_loop", physics_loops_per_inner_loop, &key_p, &val_p, a, b);
  write_opt("obs_dim", obs_dim, &key_p, &val_p, a, b);
  write_opt("act_dim", act_dim, &key_p, &val_p, a, b);
  write_opt("torque_delay_steps", torque_delay_steps, &key_p, &val_p, a, b);
  write_opt("observation_delay_steps", observation_delay_steps, &key_p, &val_p, a, b);
  write_opt("trq_dly_buflen", trq_dly_buflen, &key_p, &val_p, a, b);
  write_opt("obs_dly_buflen", obs_dly_buflen, &key_p, &val_p, a, b);
  write_opt("done_inner_steps", done_inner_steps, &key_p, &val_p, a, b);
  write_opt("neg_omega_hip_maxabs", neg_omega_hip_maxabs, &key_p, &val_p, a, b);
  write_opt("neg_omega_knee_maxabs", neg_omega_knee_maxabs, &key_p, &val_p, a, b);
  write_opt("neg_fcn_k50_co_0", neg_fcn_k50_co_0, &key_p, &val_p, a, b);
  write_opt("fc1_size", fc1_size, &key_p, &val_p, a, b);
  write_opt("fc2_size", fc2_size, &key_p, &val_p, a, b);
  write_opt("fc3_size", fc3_size, &key_p, &val_p, a, b);
  write_opt("h1_div_4", h1_div_4, &key_p, &val_p, a, b);
  write_opt("h2_div_4", h2_div_4, &key_p, &val_p, a, b);
  write_opt("mjstep1_after_mjstep", mjstep1_after_mjstep, &key_p, &val_p, a, b);
  write_opt("separate_mjstep1_mjstep2", separate_mjstep1_mjstep2, &key_p, &val_p, a, b);
  write_opt("delay_valid_obs", delay_valid_obs, &key_p, &val_p, a, b);
  write_opt("mjstep_order", mjstep_order, &key_p, &val_p, a, b);
  write_opt("rew_dly_buflen", rew_dly_buflen, &key_p, &val_p, a, b);
  write_opt("Shared_Obj", Shared_Obj, &key_p, &val_p, a, b);
  write_opt("SimIF_CPP", SimIF_CPP, &key_p, &val_p, a, b);
  write_opt("MlpIF_CPP", MlpIF_CPP, &key_p, &val_p, a, b);
  write_opt("Rollout_CPP", Rollout_CPP, &key_p, &val_p, a, b);
  write_opt("__MAINPROG__", __MAINPROG__, &key_p, &val_p, a, b);
  write_opt("do_mlp_output_tanh", do_mlp_output_tanh, &key_p, &val_p, a, b);
  write_opt("mlp_output_scaling", mlp_output_scaling, &key_p, &val_p, a, b);
  write_opt("actdim_div_2", actdim_div_2, &key_p, &val_p, a, b);
}


#if __MAINPROG__ == Rollout_CPP
  std::chrono::system_clock::time_point tm_start;
  mjtNum gettm(void)
  {
      std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - tm_start;
      return elapsed.count();
  }

  int main(int argc, char *argv[]){
    // Rollout Arguments
    const int traj_num = 1;
    const int n_steps = 8000;
    const double gamma = 0.99;

    // SimIF variables
    double theta_hip_inits[traj_num];
    double theta_knee_inits[traj_num];
    double omega_hip_inits[traj_num];
    double omega_knee_inits[traj_num];
    double pos_slider_inits[traj_num];
    double vel_slider_inits[traj_num];
    double jumping_time_inits[traj_num];
    int noise_index_inits[traj_num];

    // MLP variables
    double fc1[obs_dim * h1] = {0};
    double fc2[h1      * h2] = {0};
    double fc3[h2 * act_dim] = {0};
    double fc1_bias[h1] = {0};
    double fc2_bias[h2] = {0};
    double fc3_bias[act_dim] = {-PI / 4, -PI * 100 / 180};

    // Output variables
    double eta_greedy[traj_num];
    double return_greedy[traj_num];
    int done_steps[traj_num];

    // Scratch Variables
    int i;
    double sim_time;

    // Initializing Our Arrays
    for (i=0; i<traj_num; i++) {
      theta_hip_inits[i] = -PI * 50  / 180;
      theta_knee_inits[i] = -PI * 100 / 180;
      omega_hip_inits[i] = 0;
      omega_knee_inits[i] = 0;
      pos_slider_inits[i] = 0.4;
      vel_slider_inits[i] = 0;
      jumping_time_inits[i] = 3;
      noise_index_inits[i] = 0;
    };

    #ifdef Debug_main
      std::cout << "Step 1) Creating the Rollout" << std::endl;
      std::cout << "  --> Started Creating a Rollout instance!" << std::endl;
    #endif

    Rollout rollout = Rollout();

    #ifdef Debug_main
      std::cout << "  --> Done Creating a Rollout instance!" << std::endl;
      std::cout << "--------------------" << std::endl;
    #endif


    #ifdef Debug_main
      std::cout << "Step 2) Initializing the Rollout" << std::endl;
      std::cout << "  --> Started Initializing the Rollout's Sim Interface!" << std::endl;
    #endif

    rollout.set_simif_inits(theta_hip_inits, theta_knee_inits,
      omega_hip_inits, omega_knee_inits, pos_slider_inits, vel_slider_inits,
      jumping_time_inits, noise_index_inits);

    #ifdef Debug_main
      std::cout << "  --> Started Initializing the Rollout's MLP!" << std::endl;
    #endif

    rollout.set_mlp_weights(fc1, fc2, fc3, fc1_bias, fc2_bias, fc3_bias);

    #ifdef Debug_main
      std::cout << "  --> Done Initializing the Rollout!" << std::endl;
      std::cout << "--------------------" << std::endl;
    #endif

    #ifdef Debug_main
      std::cout << "Step 5) Starting the Greedy Simulations" << std::endl;
      std::cout << "  --> Started Rolling Trajectories!" << std::endl;
    #endif

    sim_time = gettm();
    rollout.greedy_lite(traj_num, n_steps, gamma,
                        eta_greedy, return_greedy,
                        done_steps);
    sim_time = gettm() - sim_time;

    #ifdef Debug_main
      std::cout << "  --> Done  Rolling Trajectories!" << std::endl;
      std::cout << "  --> Full Simulation Time: " << sim_time << std::endl;
      std::cout << "--------------------" << std::endl;

      std::cout << std::fixed << std::setprecision(8);
      std::cout << " The Discounted Payoff Values Are:     " << std::endl;
      for (i = 0; i < traj_num; i++)
        std::cout << eta_greedy[i] << ", ";
      std::cout << std::endl;

      std::cout << " The Non-discounted Payoff Values Are: " << std::endl;
      for (i = 0; i < traj_num; i++)
        std::cout << return_greedy[i] << ", ";
      std::cout << std::endl;

      std::cout << " The Trajectory Lengths Are:           " << std::endl;
      for (i = 0; i < traj_num; i++)
        std::cout << done_steps[i] << ", ";
      std::cout << std::endl;
    #endif

    return 0;
  }

#endif
