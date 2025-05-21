# Task-Aware Virtual Training: Enhancing Generalization in Meta-Reinforcement Learning for Out-of-Distribution Tasks (ICML2025)
This repository considers the implementation of the paper "Task-Aware Virtual Training: Enhancing Generalization in Meta-Reinforcement Learning for Out-of-Distribution Tasks" which has been accepted to ICML 2025, and is available at https://arxiv.org/abs/2502.02834. This repository is developed on top of the PEARL baseline implementation (https://github.com/katerakelly/oyster).

## Abstract 
Meta reinforcement learning aims to develop policies that generalize to unseen tasks sampled from a task distribution. While context-based meta-RL methods improve task representation using task latents, they often struggle with out-of-distribution (OOD) tasks. To address this, we propose Task-Aware Virtual Training (TAVT), a novel algorithm that accurately captures task characteristics for both training and OOD scenarios using metric-based representation learning. Our method successfully preserves task characteristics in virtual tasks and employs a state regularization technique to mitigate overestimation errors in state-varying environments. Numerical results demonstrate that TAVT significantly enhances generalization to OOD tasks across various MuJoCo and MetaWorld environments.

## Installation :
* conda environment setup is in the tavt_conda_env.yaml
* MuJoCo200 for reward function varying environments, MuJoCo131 for state transition function varying environments. MuJoCo131 library is in rand_param_envs directory so you have to do include both MuJoCo200 and MuJoCo131 binaries on the .mujoco directory and install mujoco-py 2.0.2.5 version on your environment.

## Run :

    python launch_experiment.py --env_name=cheetah-vel-ood --gpu_num=0 --exp_num=exp1 --k_model=500 --k_rl=1000 --n_meta=16 --n_vt=5 --M=3 --lambda_bisim=50 --env_mode=reward
    
    python launch_experiment.py --env_name=ant-dir-2 --gpu_num=0 --exp_num=exp1 --k_model=500 --k_rl=4000 --lambda_ent=0.5 --c_buffer_size=100000 --h_freq=100 --n_meta=2 --n_vt=1 --M=2 --env_mode=reward
    
    python launch_experiment.py --env_name=ant-dir-4 --gpu_num=0 --exp_num=exp1 --k_model=500 --k_rl=4000 --lambda_ent=0.5 --n_meta=4 --n_vt=4 --M=4 --env_mode=reward
    
    python launch_experiment.py --env_name=ant-goal-ood --gpu_num=0 --exp_num=exp1 --k_model=500 --k_rl=4000 --lambda_rew=1 --lambda_ent=0.5 --n_meta=16 --n_vt=5 --M=3 --env_mode=reward
    
    python launch_experiment.py --env_name=hopper-mass-ood --gpu_num=0 --exp_num=exp1 --k_model=1000 --k_rl=4000 --lambda_ent=0.2 --lambda_vt=0.1 --n_meta=16 --n_vt=5 --M=3 --env_mode=state
    
    python launch_experiment.py --env_name=walker-mass-ood --gpu_num=0 --exp_num=exp1 --k_model=1000 --k_rl=4000 --lambda_ent=0.2 --n_meta=16 --n_vt=5 --M=3 --env_mode=state


