"""
Launcher for experiments with PEARL

"""
import os
import pathlib
import numpy as np
import click
import json
import torch
import random

import time

import argparse

import warnings
warnings.filterwarnings("ignore")

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder, TransitionDecoder4PEARL, MyFlattenMlp
from rlkit.torch.networks import WGanCritic, Decoder, Discriminator, Autoencoder, SimpleVAE, SNWGanCritic, ReverseDynamics, AlphaNet
from rlkit.torch.networks import PsiAuxVaeDec
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config

from rlkit.torch.sac.sac_revised import PEARLSoftActorCritic

def experiment(variant):

    env = NormalizedBoxEnv(  ENVS[  variant['env_name']  ]   (**variant['env_params'])  )

    seed = variant['seed']
    print("SEED :", seed)

    env.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    tasks, total_tasks_dict_list = env.get_all_task_idx()
    print("tasks", tasks)
    print("total_tasks_dict_list[0]", total_tasks_dict_list[0])
    print("total_tasks_dict_list[1]", total_tasks_dict_list[1])
    print("total_tasks_dict_list[2]", total_tasks_dict_list[2])

    obs_dim = env.get_obs_dim()
    print("obs_dim", obs_dim)
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    print("obs_dim : {},  antion_dim : {}".format(obs_dim, action_dim))

    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] \
                                else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder



    num_train = variant["n_train_tasks"]
    num_test = variant["n_eval_tasks"]
    num_indistribution = variant["n_indistribution_tasks"]
    num_tsne = variant["n_tsne_tasks"]
    print("num_train", num_train)
    print("num_test", num_test)
    print("num_indistribution", num_indistribution)
    print("num_tsne", num_tsne)

    print("train_tasks = ", tasks[: num_train])
    print("eval_tasks = ", tasks[num_train:  num_train + num_test])
    print("indistribution_tasks = ", tasks[num_train + num_test: num_train + num_test + num_indistribution])
    print("tsne_tasks = ", tasks[num_train + num_test + num_indistribution: num_train + num_test + num_indistribution + num_tsne])


    c_distribution_vae = SimpleVAE(input_dim=latent_dim, latent_dim=latent_dim)

    psi = encoder_model(
        hidden_sizes=[300, 300, 300],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
    )
    psi_aux = encoder_model(
        hidden_sizes=[300, 300, 300],
        input_size=obs_dim + action_dim,
        output_size=context_encoder_output_dim,
    )
    psi_aux_vae_dec = PsiAuxVaeDec(latent_dim=latent_dim,
                                   obs_dim=obs_dim,
                                   action_dim=action_dim)

    use_decoder_next_state = variant['algo_params']['use_decoder_next_state']
    use_state_noise = variant['algo_params']['use_state_noise']
    print("use_state_noise", use_state_noise)
    use_target_c_dec = variant['algo_params']['use_target_c_dec']

    additional_decoder = Decoder(
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_dim=1,
        use_next_state=use_decoder_next_state,
        use_k=True,
        num_tasks=num_train,
        use_state_noise=use_state_noise
    )
    task_decoder = Decoder(
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_dim=1,
        use_next_state=use_decoder_next_state,
        use_k=False,
        num_tasks=num_train,
        use_state_noise=use_state_noise
    )

    task_decoder_target = Decoder(
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_dim=1,
        use_next_state=use_decoder_next_state,
        use_k=False,
        num_tasks=num_train,
        use_state_noise=use_state_noise
    )

    if variant['algo_params']['gan_type'] == 'wgan':
        disc_model = WGanCritic
    elif variant['algo_params']['gan_type'] == 'sngan':
        disc_model = SNWGanCritic
    else:
        disc_model = None
    disc_l_dim = latent_dim if variant['algo_params']['use_latent_in_disc'] else 0

    discriminator = disc_model(
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_dim=1,
        latent_dim=disc_l_dim)

    print("additional_decoder", additional_decoder)
    print("task_decoder", task_decoder)
    print("discriminator", discriminator)

    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim if not variant["algo_params"]["use_index_rl"] else obs_dim + num_train,
        latent_dim=latent_dim if not variant["algo_params"]["use_index_rl"] else num_train,
        action_dim=action_dim,
    )


    log_alpha_net = AlphaNet(latent_dim=latent_dim)

    agent = PEARLAgent(
        latent_dim,
        psi,
        psi_aux_vae_dec,
        additional_decoder,
        task_decoder,
        task_decoder_target,
        discriminator,
        policy,
        log_alpha_net,
        c_distribution_vae,
        tasks[: num_train],
        **variant['algo_params']
    )

    exp_name = variant['exp_name']
    launch_file_name = os.path.basename(__file__)
    print("__file__", __file__)
    print("launch_file_name = os.path.basename(__file__)", launch_file_name)
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_name,
                                      base_log_dir=variant['util_params']['base_log_dir'])


    algorithm = PEARLSoftActorCritic(
        env=env,
        train_tasks=tasks[: num_train],
        eval_tasks=tasks[num_train:  num_train + num_test],
        indistribution_tasks=tasks[num_train + num_test: num_train + num_test + num_indistribution],
        tsne_tasks=tasks[num_train + num_test + num_indistribution: num_train + num_test + num_indistribution + num_tsne],
        total_tasks_dict_list=total_tasks_dict_list,
        nets=[agent, qf1, qf2, vf],
        latent_dim=latent_dim,
        log_dir=experiment_log_dir,
        launch_file_name=launch_file_name,
        dims={"obs_dim":obs_dim, "action_dim":action_dim, "reward_dim":1, "latent_dim": latent_dim},
        exp_name=exp_name,
        config=variant,
        **variant['algo_params']
    )


    if ptu.gpu_enabled():
        algorithm.to()

    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    algorithm.train()

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to



def main(args):


    variant = default_config
    config = "configs/" + args.env_name + ".json"
    if config:

        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)

        variant["algo_params"]["beta"] = args.beta

        # variant["algo_params"]["pretrain_steps"] = args.pretrain_steps
        variant["algo_params"]["k_model"] = args.k_model
        variant["algo_params"]["k_rl"] = args.k_rl
        variant["algo_params"]["num_initial_steps"] = args.num_initial_steps

        variant["algo_params"]["lambda_rew"] = args.lambda_rew

        variant["algo_params"]["target_lambda_ent"] = args.target_lambda_ent



        variant["algo_params"]["lambda_recon"] = args.lambda_recon
        variant["algo_params"]["lambda_same_task"] = args.lambda_same_task
        variant["algo_params"]["lambda_onoff"] = args.lambda_onoff
        variant["algo_params"]["lambda_offoff"] = args.lambda_offoff
        variant["algo_params"]["lambda_bisim"] = args.lambda_bisim
        variant["algo_params"]["bisim_penalty_coeff"] = args.bisim_penalty_coeff
        variant["algo_params"]["lambda_tp"] = args.lambda_tp
        variant["algo_params"]["r_dist_coeff"] = args.r_dist_coeff
        variant["algo_params"]["eta"] = args.eta

        variant["algo_params"]["lambda_wgan"] = args.lambda_wgan


        variant["algo_params"]["gen_freq"] = args.gen_freq

        variant["algo_params"]["lambda_gp"] = args.lambda_gp

        variant["algo_params"]["c_kl_lambda"] = args.c_kl


        variant["algo_params"]["c_distri_vae_train_freq"] = args.c_vae_freq
        variant["algo_params"]["fakesample_rl_tran_batch_size"] = args.f_bsize

        # variant["algo_params"]["which_sac_file"] = args.which_sac_file

        variant["algo_params"]["gan_type"] = args.gan_type

        if 'goal' in args.env_name:
            if args.contact == "True":
                variant["env_params"]["use_cfrc"] = True
            else:
                variant["env_params"]["use_cfrc"] = False

        if args.use_c_dist_clear == "True":
            variant["algo_params"]["use_c_dist_clear"] = True
        elif args.use_c_dist_clear == "False":
            variant["algo_params"]["use_c_dist_clear"] = False

        if args.use_gan == "True":
            variant["algo_params"]["use_gan"] = True
        elif args.use_gan == "False":
            variant["algo_params"]["use_gan"] = False

        if args.fakesample_cycle == "True":
            variant["algo_params"]["fakesample_cycle"] = True
        elif args.fakesample_cycle == "False":
            variant["algo_params"]["fakesample_cycle"] = False

        if args.use_latent_in_disc == "True":
            variant["algo_params"]["use_latent_in_disc"] = True
        elif args.use_latent_in_disc == "False":
            variant["algo_params"]["use_latent_in_disc"] = False

        if args.use_vt_rl == "True":
            variant["algo_params"]["use_vt_rl"] = True
        elif args.use_vt_rl == "False":
            variant["algo_params"]["use_vt_rl"] = False

        if args.use_first_samples_for_bisim_samples == "True":
            variant["algo_params"]["use_first_samples_for_bisim_samples"] = True
        elif args.use_first_samples_for_bisim_samples == "False":
            variant["algo_params"]["use_first_samples_for_bisim_samples"] = False

        if args.use_c_vae == "True":
            variant["algo_params"]["use_c_vae"] = True
        elif args.use_c_vae == "False":
            variant["algo_params"]["use_c_vae"] = False

        if args.use_new_batch_for_fake == "True":
            variant["algo_params"]["use_new_batch_for_fake"] = True
        elif args.use_new_batch_for_fake == "False":
            variant["algo_params"]["use_new_batch_for_fake"] = False

        if args.use_penalty == "True":
            variant["algo_params"]["use_penalty"] = True
        elif args.use_penalty == "False":
            variant["algo_params"]["use_penalty"] = False

        if args.use_vt_representation == "True":
            variant["algo_params"]["use_vt_representation"] = True
        elif args.use_vt_representation == "False":
            variant["algo_params"]["use_vt_representation"] = False

        if args.use_fake_value_bound == "True":
            variant["algo_params"]["use_fake_value_bound"] = True
        elif args.use_fake_value_bound == "False":
            variant["algo_params"]["use_fake_value_bound"] = False

        if args.use_episodic_online_buffer == "True":
            variant["algo_params"]["use_episodic_online_buffer"] = True
        elif args.use_episodic_online_buffer == "False":
            variant["algo_params"]["use_episodic_online_buffer"] = False

        if args.c_off_all_element_sampling == "True":
            variant["algo_params"]["c_off_all_element_sampling"] = True
        elif args.c_off_all_element_sampling == "False":
            variant["algo_params"]["c_off_all_element_sampling"] = False

        if args.use_c_off_rl == "True":
            variant["algo_params"]["use_c_off_rl"] = True
        elif args.use_c_off_rl == "False":
            variant["algo_params"]["use_c_off_rl"] = False

        if args.use_index_rl == "True":
            variant["algo_params"]["use_index_rl"] = True
        elif args.use_index_rl == "False":
            variant["algo_params"]["use_index_rl"] = False

        if args.use_next_obs_Q_reg == "True":
            variant["algo_params"]["use_next_obs_Q_reg"] = True
        elif args.use_next_obs_Q_reg == "False":
            variant["algo_params"]["use_next_obs_Q_reg"] = False

        if args.use_auto_entropy == "True":
            variant["algo_params"]["use_auto_entropy"] = True
        elif args.use_auto_entropy == "False":
            variant["algo_params"]["use_auto_entropy"] = False

        if args.use_target_c_dec == "True":
            variant["algo_params"]["use_target_c_dec"] = True
        elif args.use_target_c_dec == "False":
            variant["algo_params"]["use_target_c_dec"] = False

        # if args.use_epsilon_reg == "True":
        #     variant["algo_params"]["use_epsilon_reg"] = True
        # elif args.use_epsilon_reg == "False":
        #     variant["algo_params"]["use_epsilon_reg"] = False

        if args.use_rewards_beta == "True":
            variant["algo_params"]["use_rewards_beta"] = True
        elif args.use_rewards_beta == "False":
            variant["algo_params"]["use_rewards_beta"] = False

        if args.use_sample_reg == "True":
            variant["algo_params"]["use_sample_reg"] = True
        elif args.use_sample_reg == "False":
            variant["algo_params"]["use_sample_reg"] = False

        if args.use_tp_detach == "True":
            variant["algo_params"]["use_tp_detach"] = True
        elif args.use_tp_detach == "False":
            variant["algo_params"]["use_tp_detach"] = False

        if args.wide_exp == "True":
            variant["algo_params"]["wide_exp"] = True
        elif args.wide_exp == "False":
            variant["algo_params"]["wide_exp"] = False

        if args.use_closest_task == "True":
            variant["algo_params"]["use_closest_task"] = True
        elif args.use_closest_task == "False":
            variant["algo_params"]["use_closest_task"] = False


        variant["algo_params"]["optimizer"] = args.optimizer

        variant["algo_params"]["sample_reg_method"] = args.sample_reg_method
        variant["algo_params"]["epsilon_reg"] = args.epsilon_reg
        variant["algo_params"]["rewards_beta"] = args.rewards_beta

        variant["algo_params"]["q_reg_coeff"] = args.q_reg_coeff

        variant["algo_params"]["c_off_batch_size"] = args.c_off_batch_size

        variant["algo_params"]["bisim_transition_sample"] = args.bisim_transition_sample

        variant["algo_params"]["c_buffer_size"] = args.c_buffer_size

        variant["algo_params"]["lambda_ent"] = args.lambda_ent

        variant["algo_params"]["h_freq"] = args.h_freq

        variant["algo_params"]["wandb_project"] = args.wandb_project

        variant["algo_params"]["closest_task_method"] = args.closest_task_method
        variant['algo_params']['close_method'] = args.close_method

        variant["seed"] = args.seed

        variant["algo_params"]["alpha_net_weight_decay"] = args.alpha_net_weight_decay

        variant["algo_params"]["psi_aux_vae_beta"] = args.psi_aux_vae_beta

        variant["algo_params"]["lambda_vt"] = args.lambda_vt
        variant['util_params']['gpu_id'] =  args.gpu_num


        # if args.use_manual_tasks_sampling == "True":
        variant["algo_params"]["n_meta"] = args.n_meta
        variant["algo_params"]["n_vt"] = args.n_vt
        variant["algo_params"]["M"] = args.M

        variant["exp_name"] = args.exp_num

        if args.env_mode == "reward":
            variant["algo_params"]["use_decoder_next_state"] = False
            variant["algo_params"]["use_next_obs_in_context"] = False
            variant["algo_params"]["use_next_state_bisim"] = False
            variant["algo_params"]["use_state_noise"] = False
            variant["algo_params"]["use_epsilon_reg"] = False
        elif args.env_mode == "state":
            variant["algo_params"]["use_decoder_next_state"] = True
            variant["algo_params"]["use_next_obs_in_context"] = True
            variant["algo_params"]["use_next_state_bisim"] = True
            variant["algo_params"]["use_state_noise"] = True
            variant["algo_params"]["use_epsilon_reg"] = True




        ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])


    experiment(variant)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_name', default='ant-goal-ood', type=str)  # Requirement
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--gpu_num', default=0, type=int)
    parser.add_argument('--exp_num', default="1", type=str)
    parser.add_argument('--k_model', default=500, type=int)
    parser.add_argument('--k_rl', default=4000, type=int)
    parser.add_argument('--lambda_rew', default=5.0, type=float)
    parser.add_argument('--lambda_ent', default=1.0, type=float)
    parser.add_argument('--beta', default=2.0, type=float)
    parser.add_argument('--n_meta', default=0, type=int)  # Requirement
    parser.add_argument('--n_vt', default=0, type=int)  # Requirement
    parser.add_argument('--M', default=0, type=int)  # Requirement
    parser.add_argument('--eta', default=0.1, type=float)
    # parser.add_argument('--use_epsilon_reg', default="False", type=str)
    parser.add_argument('--epsilon_reg', default=0.1, type=float)
    parser.add_argument('--gan_type', default="wgan", type=str)
    parser.add_argument('--h_freq', default=20, type=int)
    parser.add_argument('--lambda_recon', default=200, type=float)
    parser.add_argument('--lambda_onoff', default=100, type=float)
    parser.add_argument('--lambda_offoff', default=100, type=float)
    parser.add_argument('--lambda_same_task', default=0, type=float)
    parser.add_argument('--lambda_tp', default=100, type=int)
    parser.add_argument('--lambda_bisim', default=100, type=int)
    parser.add_argument('--lambda_wgan', default=1.0, type=float)
    parser.add_argument('--lambda_gp', default=5, type=float)
    parser.add_argument('--lambda_vt', default=1, type=float)
    parser.add_argument('--gen_freq', default=5, type=int)
    parser.add_argument('--c_buffer_size', default=5000, type=int)

    parser.add_argument('--use_vt_representation', default="True", type=str)
    parser.add_argument('--use_vt_rl', default="True", type=str)
    parser.add_argument('--use_latent_in_disc', default="True", type=str)
    parser.add_argument('--use_tp_detach', default="True", type=str)

    parser.add_argument('--env_mode', default="reward", type=str)  # choose(reward, state)





    parser.add_argument('--use_auto_entropy', default="False", type=str)
    parser.add_argument('--optimizer', default="adam", type=str)
    parser.add_argument('--num_initial_steps', default=2000, type=int)
    parser.add_argument('--use_c_vae', default="False", type=str)
    parser.add_argument('--bisim_transition_sample', default="rl", type=str)
    parser.add_argument('--use_closest_task', default="False", type=str)
    parser.add_argument('--closest_task_method', default="total", type=str)
    parser.add_argument('--close_method', default="old", type=str)
    parser.add_argument('--c_off_all_element_sampling', default="False", type=str)
    parser.add_argument('--use_target_c_dec', default="False", type=str)
    parser.add_argument('--use_c_off_rl', default="False", type=str)
    parser.add_argument('--use_index_rl', default="False", type=str)
    parser.add_argument('--use_fake_value_bound', default="False", type=str)
    parser.add_argument('--use_first_samples_for_bisim_samples', default="True", type=str)
    parser.add_argument('--alpha_net_weight_decay', default=0.0, type=float)
    parser.add_argument('--target_lambda_ent', default=1.0, type=float)
    parser.add_argument('--contact', default="True", type=str)
    parser.add_argument('--sample_dist_coeff', default=1, type=float)
    parser.add_argument('--r_dist_coeff', default=1, type=float)

    parser.add_argument('--use_rewards_beta', default="False", type=str)
    parser.add_argument('--rewards_beta', default=1.0, type=float)
    parser.add_argument('--sample_reg_method', default="", type=str)
    parser.add_argument('--use_sample_reg', default="False", type=str)
    parser.add_argument('--wide_exp', default="False", type=str)
    parser.add_argument('--bisim_penalty_coeff', default=1.0, type=float)
    parser.add_argument('--q_reg_coeff', default=1.0, type=float)
    parser.add_argument('--c_off_batch_size', default=32, type=int)
    parser.add_argument('--use_gan', default="True", type=str)
    parser.add_argument('--use_penalty', default="False", type=str)
    parser.add_argument('--use_c_dist_clear', default="True", type=str)
    parser.add_argument('--use_episodic_online_buffer', default="True", type=str)
    parser.add_argument('--use_next_obs_Q_reg', default="False", type=str)
    parser.add_argument('--use_new_batch_for_fake', default="True", type=str)
    parser.add_argument('--c_kl', default=0.05, type=float)
    parser.add_argument('--c_vae_freq', default=2, type=int)
    parser.add_argument('--f_bsize', default=256, type=int)
    parser.add_argument('--psi_aux_vae_beta', default=2.0, type=float)
    parser.add_argument('--fakesample_cycle', default="True", type=str)
    parser.add_argument('--wandb_project', default="tavt_repro_2", type=str)
    # parser.add_argument('--use_manual_tasks_sampling', default="False", type=str)




    args, rest_args = parser.parse_known_args()

    main(args)

