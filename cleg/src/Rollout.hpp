#pragma once

#include "mujoco.h"
#include "defs.hpp"
#include "SimIF.hpp"
#include "MlpIF.hpp"

class Rollout{
  public:
    Rollout();                         // This is the constructor
    //~Rollout();                      // This is the destructor

    void set_simif_inits(double* theta_hip, double* theta_knee,
      double* omega_hip, double* omega_knee, double* pos_slider, double* vel_slider,
      double* jumping_time, int* noise_index){
      theta_hip_inits = theta_hip;
      theta_knee_inits = theta_knee;
      omega_hip_inits = omega_hip;
      omega_knee_inits = omega_knee;
      pos_slider_inits = pos_slider;
      vel_slider_inits = vel_slider;
      jumping_time_inits = jumping_time;
      noise_index_inits = noise_index;
    };

    void set_mlp_weights(double* fc1, double* fc2, double* fc3,
      double* fc1_bias, double* fc2_bias, double* fc3_bias){
      net.set_weights(fc1, fc2, fc3, fc1_bias, fc2_bias, fc3_bias);
    };

    void greedy_lite(int traj_num, int n_steps, double gamma,
                     double* eta_greedy, double* return_greedy,
                     int* done_steps);

    void vine_lite(int traj_num, int n_steps, double gamma,
                   int expl_steps, int* reset_times, double* expl_noise,
                   double* obs_greedy, double* action_greedy, double* action_vine,
                   double* Q_greedy, double* eta_greedy, double* return_greedy,
                   double* Q_vine, double* eta_vine, double* return_vine,
                   int* done_steps, int* done_steps_vine);

    void stochastic(int traj_num, int n_steps, double* expl_noise,
                    double* obs, double* action, double* rewards,
                    int* done_steps);

    void reset(int traj_idx);
    void partial_stochastic(int n_steps, double* expl_noise, double* obs,
                            double* action, double* rewards, bool* dones);


    void infer_mlp(int input_num, double* mlp_input, double* mlp_output) {
      for (int i = 0; i < input_num; i++){
        mlp_action = net.forward(mlp_input + obs_dim * i);
        mju_copy(mlp_output + act_dim * i, mlp_action, act_dim);
      }
    }

  private:
    SimInterface simiface;
    MLP3 net;

    // Sim Interface Initialization Arguments (Arrays)
    double* theta_hip_inits;
    double* theta_knee_inits;
    double* omega_hip_inits;
    double* omega_knee_inits;
    double* pos_slider_inits;
    double* vel_slider_inits;
    double* jumping_time_inits;
    int* noise_index_inits;

    // A bunch of scratch variables
    int traj_idx, step;
    double* mlp_action;
    double* observation;
    double reward;
    bool done;
    double gamma_pow;

    // vine_lite scratch Variables
    double* eta;
    double* return_;
    double* Q;
    int gen_idx;
    int obs_g_idx;
    int act_idx;
    int st_expl;
    int end_expl;
    double gamma_pow_q;
};

void write_opt(const std::string key_str, double val, char** key_write_ptr,
               double** val_write_ptr, char* key_write_ptr_max,
               double* val_write_ptr_max);

void get_build_options(char* keys, double* vals, int keys_len, int vals_len);
