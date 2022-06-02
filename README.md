# Truly Deterministic Policy Optimization

[Truly Deterministic Policy Optimization (TDPO)](https://arxiv.org/abs/2205.15379) is a model-free policy gradient method which trains reinforcement learning agents without requiring a Gaussian stochasticity in the policy's action. This is why we call it truly deterministic; not only it uses *deterministic policy gradients*, but also it performs *deterministic policy search*. Notice the distinction between *deterministic search* and *deterministic gradients*; DDPG and TD3 use deterministic policy gradients, but they still inject Ornstein or Gaussian noises and thus perform stochastic policy search. To the best our our knowledge, our work is the first to practically implement such a *deterministic policy search* strategy.

Deterministic policy search has many potential merits.
* First of all, it can reduce the estimation variance in existing policy gradient methods; less training noise injection can certainly reduce the variance and inconsistencies.
* Furthermore, it can make trainings for longer episodes more practical. The *curse of horizon* has been under-stressed in modern reinforcement learning. Deterministic policy search may unlock the ability to train environments with considerably larger discount factors and episode lengths.
* Also, it may be more resilient to non-MDP artifacts (such as non-local rewards, observation or action delays, etc.).
* Finally, it can be a valuable asset to safe-RL; it allows training RL methods in noise-susceptible environments, where injecting Gaussian or Ornstein noises can be harmful to the device.

To show the practicality of our approach, we tested the TDPO agents on hardware to control a leg of the [MIT Cheetah robot](https://www.youtube.com/watch?v=_luhn7TLfWU). The MIT Cheetah is a high-performance device, quite challenging to control. For higher power, this robot lacks spring dampers and it can exert more than 50 N.m. of torque at 30 radians per second velocity. This is more than enough to break a human hand in a split second if the controller is not precise enough. None of the existing model-free RL methods could perform global control on this device, even in simulation, after years of performing systematic hyper-parameter optimization and even code-level optimizations. This lack of practical methods for challenging environments motivated the design of TDPO.

Here is a one minute physical test demo of the TDPO agent performing drop-and-catch tests on this leg at 4kHz frequency, and smoothly recovering from 70 cm drops:



https://user-images.githubusercontent.com/28830570/171543989-ce40e5ad-4827-4a5d-a053-0de6861583a2.mp4



## Training Details

* We used Python 3.6, and the exact library versions are included in the `requirements.txt` file:

```bash
python -m pip install -r requirements.txt
```

* You also need to install [Mujoco](https://mujoco.org/). It's open-source now.

* To train the TDPO agent on the leg environment, run

```bash
./train.sh
```

* To run the other methods (`ppo1`, `ppo2`, `trpo`, `td3`, etc.), you need to specify the config files:
  - All of our config files are given in the [`./bench/configs`](./bench/configs) directory as json files.
  - All the hyper-parameter optimization configs for the test benchmark are included in [`./bench/configs/3_lfhpo_west`](./bench/configs/3_lfhpo_west)
    - The One-Variable-at-a-Time (OVAT) parameter sweep configurations for the test benchmark are given in the following files:
      - For `ppo1`, see [`./bench/configs/3_lfhpo_west/0_ppo1_ovat.json`](./bench/configs/3_lfhpo_west/0_ppo1_ovat.json)
      - For `trpo`, see [`./bench/configs/3_lfhpo_west/0_trpo_ovat.json`](./bench/configs/3_lfhpo_west/0_trpo_ovat.json)
      - For `td3`, see [`./bench/configs/3_lfhpo_west/0_td3_ovat.json`](./bench/configs/3_lfhpo_west/0_td3_ovat.json)
    - At [`./bench/configs3_lfhpo_west/1_trpo_optuna/hpo.json`](./bench/configs3_lfhpo_west/1_trpo_optuna/hpo.json) you can find the hyper-parameter optimization configs we used for Optuna on TRPO.
      - The configs proposed by Optuna for TRPO on the test problem are also placed next to it. The files with a `round_*.json` name contain the TRPO configs proposed by Optuna.
      - For instance, [`./bench/configs3_lfhpo_west/1_trpo_optuna/round_0.json`](./bench/configs3_lfhpo_west/1_trpo_optuna/round_0.json) has the first set of proposed hyper-parameters by Optuna for TRPO.
    - The other HPO configs and proposed hyper-parameters for GPyOpt, BayesianOptimization, Scikit-Optimize, Optuna, and ProSRS are also given [`./bench/configs3_lfhpo_west/1_trpo_optuna`](./bench/configs3_lfhpo_west/1_trpo_optuna)
  - The OVAT parameter sweep JSON configs for the long-horizon test benchmark are given at the [`./bench/configs/2_ovat5b_eng`](./bench/configs/2_ovat5b_eng) directory.
  - The HPO configs and proposed hyper-parameters are given at the [`./bench/configs/4_hpo5b_eng`](./bench/configs/4_hpo5b_eng) directory.
* Once you selected your specific config files to run, you can run them using either of two ways:
  - **Multi-Node SLURM Job Submission**: If you want to use our SLURM scripts to distribute the job among many nodes on a computing cluster, feel free to use the [`./bench/jobscripts/sbatch.sh`](./bench/jobscripts/sbatch.sh) script. You only need to specify your config inside the `CFGPREFIXARR` variable, and it will submit the job for you.
  - **Single Node Runs from a Shell Terminal**: If you want to run the config from a shell script on a single node, take a look at [`./bench/trainer.sh`](./bench/trainer.sh) shell script, and specify the config file in the `CFGPREFIX` variable.

This is the bare minimum information to use the repository. Of course, we will be updating the repository with better documentation and more user-friendly scripts. If you have trouble setting up the environment and libraries or when running the code, please don't hesitate to reach out to us either at `ehsans2@illinois.edu` or make an issue here.

# References

You can find our paper at [https://arxiv.org/abs/2205.15379](https://arxiv.org/abs/2205.15379).

Here's the bibtex citation for our work:

```
@misc{saleh2022truly,
      title={Truly Deterministic Policy Optimization},
      author={Ehsan Saleh and Saba Ghaffari and Timothy Bretl and Matthew West},
      year={2022},
      eprint={2205.15379},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
