import abc
from collections import OrderedDict
import time

import gtimer as gt
import numpy as np
import os
import shutil

import torch
from torch.distributions.dirichlet import Dirichlet

from rlkit.core import logger, eval_util
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers.in_place import InPlacePathSampler
from rlkit.torch import pytorch_util as ptu

# from MulticoreTSNE import MulticoreTSNE as TSNE
# pip install git+https://github.com/jorvis/Multicore-TSNE


import matplotlib.pyplot as plt

# import wandb
import datetime

import random
from collections import deque


class EpisodicOnlineReplayBuffer:
    def __init__(self, num_task, max_episode_per_buffer=10):
        self.num_task = num_task
        self.max_episode_per_buffer = max_episode_per_buffer
        self.buffers = [deque(maxlen=max_episode_per_buffer) for i in range(num_task)]

    def add_context(self, task_index, context):
        self.buffers[task_index].append(context)

    def sample_context(self, task_indices, batch_size):
        batches = []
        for idx in task_indices:
            random_epi_context_in_buffer = random.sample(self.buffers[idx], 1)[0]
            rand_tran_indices = np.random.choice(range(random_epi_context_in_buffer.size(1)), batch_size)
            batches.append(random_epi_context_in_buffer[:, rand_tran_indices, :])
        batches = torch.cat(batches)
        return batches


class MetaRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            agent,
            train_tasks,
            eval_tasks,
            indistribution_tasks,
            tsne_tasks,
            total_tasks_dict_list,

            use_state_noise=True,
            use_new_batch_4_fake=False,

            ood="inter",
            env_name_suffix="",

            lambda_recon=200,
            lambda_onoff=100,
            lambda_offoff=100,
            lambda_same_task=200,
            lambda_tp=10,
            lambda_wgan=1,

            bisim_penalty_coeff=1,

            which_sac_file='',

            use_full_interpolation=True,

            lambda_bisim=50,

            M=4,
            n_vt=5,

            r_dist_coeff=1,
            eta=1,
            policy_kl_reg_coeff=0,
            pretrain_tsne_freq=5000,

            use_c_dist_clear=True,

            sa_perm=False,

            gan_type=None,
            use_gan=False,

            make_prior_to_rl=False,
            fakesample_cycle=False,

            use_decrease_mask=False,
            use_vt_representation=False,

            offpol_ctxt_sampling_buffer="rl",

            c_distri_vae_train_freq=50,

            fakesample_rl_tran_batch_size=128,

            sample_dist_coeff=1,
            use_decoder_next_state=False,
            use_next_state_bisim=False,

            c_kl_lambda=0.1,

            decrease_rate=2,

            bisim_r_coef=1,
            bisim_dist_coef=1,

            c_buffer_size=2000,
            c_batch_num=500,

            beta=2,

            use_z_autoencoder=False,

            z_dist_compute_method="euclidian",
            env_name="cheetah-vel",

            algorithm="ours",

            same_task_loss_pow=1,

            clear_enc_buffer=1,
            prior_enc_buffer_size=50000,
            online_enc_buffer_size=10000,

            pretrain_steps=50000,

            lambda_gp=10,
            gen_freq=5,

            use_context_buffer=1,

            use_W=1,

            use_q_contrastive=0,
            q_embed_size=20,

            enc_q_recover_train=1,

            use_z_contrastive=1,

            use_new_batch_for_fake=True,

            target_enc_tau=0.005,

            use_c_vae=True,

            num_tsne_evals=10,
            tsne_plot_freq=5,
            tsne_perplexity=[50, 50],

            k_model=200,

            use_vt_rl=0,
            lambda_vt=0.01,

            n_meta=64,
            num_iterations=100,
            k_rl=1000,
            num_initial_steps=100,
            num_tasks_sample=100,
            num_steps_prior=100,
            num_steps_posterior=100,
            num_extra_rl_steps_posterior=100,
            num_evals=10,
            num_steps_per_eval=1000,
            batch_size=1024,
            embedding_batch_size=1024,
            embedding_mini_batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size=1000000,
            lambda_rew=1,
            num_exp_traj_eval=1,
            update_post_train=1,
            eval_deterministic=True,
            render=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=False,
            render_eval_paths=False,
            dump_eval_paths=False,
            plotter=None,

            use_latent_in_disc=True,
            use_penalty=False,
            psi_aux_vae_beta=1,

            h_freq=10,
            bisim_transition_sample='rl',
            use_fake_value_bound=False,
            wandb_project='',
            use_episodic_online_buffer=True,
            c_off_batch_size=32,
            c_off_all_element_sampling=False,
            use_c_off_rl=False,
            use_index_rl=False,
            use_first_samples_for_bisim_samples=True,
            use_next_obs_Q_reg=False,
            q_reg_coeff=1,
            use_auto_entropy=False,
            use_target_c_dec=False,
            use_epsilon_reg=False,
            epsilon_reg=1,
            use_kl_penalty=False,
            pi_kl_reg_coeff=0,
            task_mix_freq=1,
            use_rewards_beta=False,
            rewards_beta=1,
            use_sample_reg=False,
            sample_reg_method="",
            target_lambda_ent=1,
            use_tp_detach=False,
            seed=1234,
            wide_exp=False,
            optimizer='adam',
            alpha_net_weight_decay=0.0,
            lambda_ent=1.0,
            close_method="",
            use_closest_task=False,
            closest_task_method="total",

            log_dir='',
            launch_file_name='',
            dims={},
            exp_name="",
            config={}
    ):
        """
        :param env: training env
        :param agent: agent that is conditioned on a latent variable z that rl_algorithm is responsible for feeding in
        :param train_tasks: list of tasks used for training
        :param eval_tasks: list of tasks used for eval

        see default experiment config file for descriptions of the rest of the arguments
        """
        self.use_closest_task = use_closest_task
        self.closest_task_method = closest_task_method
        self.close_method = close_method
        self.lambda_ent = lambda_ent
        self.alpha_net_weight_decay = alpha_net_weight_decay
        self.optimizer = optimizer
        self.seed = seed
        self.wide_exp = wide_exp
        self.use_tp_detach = use_tp_detach
        self.target_lambda_ent = target_lambda_ent
        self.sample_reg_method = sample_reg_method
        self.use_sample_reg = use_sample_reg
        self.rewards_beta = rewards_beta
        self.use_rewards_beta = use_rewards_beta
        self.task_mix_freq = task_mix_freq
        self.use_kl_penalty = use_kl_penalty
        self.pi_kl_reg_coeff = pi_kl_reg_coeff
        self.epsilon_reg = epsilon_reg
        self.use_epsilon_reg = use_epsilon_reg
        self.config = config
        self.use_target_c_dec = use_target_c_dec
        self.q_reg_coeff = q_reg_coeff
        self.use_next_obs_Q_reg = use_next_obs_Q_reg
        self.use_first_samples_for_bisim_samples = use_first_samples_for_bisim_samples
        self.use_index_rl = use_index_rl
        self.use_c_off_rl = use_c_off_rl
        self.c_off_all_element_sampling = c_off_all_element_sampling
        self.c_off_batch_size = c_off_batch_size
        self.use_episodic_online_buffer = use_episodic_online_buffer
        self.wandb_project = wandb_project
        self.use_fake_value_bound = use_fake_value_bound
        self.bisim_transition_sample = bisim_transition_sample
        self.use_vt_representation = use_vt_representation
        self.h_freq = h_freq
        self.bisim_penalty_coeff = bisim_penalty_coeff
        self.psi_aux_vae_beta = psi_aux_vae_beta
        self.use_penalty = use_penalty
        self.use_latent_in_disc = use_latent_in_disc
        self.env = env
        self.agent = agent
        self.exploration_agent = agent
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.indistribution_tasks = indistribution_tasks
        self.tsne_tasks = tsne_tasks
        self.total_tasks_dict_list = total_tasks_dict_list

        self.lambda_onoff = lambda_onoff
        self.lambda_offoff = lambda_offoff
        self.lambda_same_task = lambda_same_task
        self.lambda_tp = lambda_tp

        curr_time = datetime.datetime.now()
        month = "0" + str(curr_time.month) if len(str(curr_time.month)) < 2 else str(curr_time.month)  # 04
        day = "0" + str(curr_time.day) if len(str(curr_time.day)) < 2 else str(curr_time.day)  # 16
        self.date = month + day

        self.use_new_batch_for_fake = use_new_batch_for_fake

        self.which_sac_file = which_sac_file

        self.lambda_bisim = lambda_bisim
        self.lambda_wgan = lambda_wgan

        self.lambda_recon = lambda_recon

        print("self.train_tasks", self.train_tasks)
        print("self.eval_tasks", self.eval_tasks)
        print("self.indistribution_tasks", self.indistribution_tasks)
        print("self.tsne_tasks", self.tsne_tasks)

        self.use_c_vae = use_c_vae

        self.use_new_batch_4_fake = use_new_batch_4_fake

        self.ood = ood
        self.env_name_suffix = env_name_suffix

        self.use_full_interpolation = use_full_interpolation

        self.use_c_dist_clear = use_c_dist_clear

        self.r_dist_coeff = r_dist_coeff
        self.eta = eta
        self.policy_kl_reg_coeff = policy_kl_reg_coeff
        self.pretrain_tsne_freq = pretrain_tsne_freq

        self.sa_perm = sa_perm

        self.n_vt = n_vt
        self.gan_type = gan_type
        self.use_gan = use_gan

        self.M = M

        self.fakesample_cycle = fakesample_cycle

        self.use_decrease_mask = use_decrease_mask

        self.make_prior_to_rl = make_prior_to_rl

        self.offpol_ctxt_sampling_buffer = offpol_ctxt_sampling_buffer

        self.c_distri_vae_train_freq = c_distri_vae_train_freq

        self.fakesample_rl_tran_batch_size = fakesample_rl_tran_batch_size

        self.sample_dist_coeff = sample_dist_coeff

        self.use_decoder_next_state = use_decoder_next_state
        self.use_next_state_bisim = use_next_state_bisim

        self.c_kl_lambda = c_kl_lambda

        self.bisim_r_coef = bisim_r_coef
        self.bisim_dist_coef = bisim_dist_coef

        self.exp_name = exp_name

        self.beta = beta
        self.c_buffer_size = c_buffer_size
        self.c_batch_num = c_batch_num

        self.decrease_rate = decrease_rate

        self.use_z_autoencoder = use_z_autoencoder

        self.same_task_loss_pow = same_task_loss_pow

        self.z_dist_compute_method = z_dist_compute_method

        self.env_name = env_name

        self.clear_enc_buffer = clear_enc_buffer
        self.prior_enc_buffer_size = prior_enc_buffer_size
        self.online_enc_buffer_size = online_enc_buffer_size

        self.pretrain_steps = pretrain_steps

        self.lambda_gp = lambda_gp
        self.gen_freq = gen_freq

        self.use_context_buffer = use_context_buffer

        self.use_W = use_W

        self.use_q_contrastive = use_q_contrastive
        self.q_embed_size = q_embed_size
        self.enc_q_recover_train = enc_q_recover_train

        self.use_z_contrastive = use_z_contrastive

        self.target_enc_tau = target_enc_tau

        self.num_tsne_evals = num_tsne_evals
        self.tsne_plot_freq = tsne_plot_freq
        self.tsne_perplexity = tsne_perplexity

        self.k_model = k_model

        self.use_vt_rl = use_vt_rl
        self.lambda_vt = lambda_vt

        self.n_meta = n_meta
        self.num_iterations = num_iterations
        self.k_rl = k_rl
        self.num_initial_steps = num_initial_steps
        self.num_tasks_sample = num_tasks_sample
        self.num_steps_prior = num_steps_prior
        self.num_steps_posterior = num_steps_posterior
        self.num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self.num_evals = num_evals
        self.num_steps_per_eval = num_steps_per_eval
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        self.embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.lambda_rew = lambda_rew
        self.update_post_train = update_post_train
        self.num_exp_traj_eval = num_exp_traj_eval
        self.eval_deterministic = eval_deterministic
        self.render = render
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment

        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.dump_eval_paths = dump_eval_paths
        self.plotter = plotter

        self.launch_file_name = launch_file_name

        self.log_dir = log_dir
        shutil.copytree("configs", os.path.join(self.log_dir, "save_files", "configs"))
        shutil.copytree("rand_param_envs", os.path.join(self.log_dir, "save_files", "rand_param_envs"))
        shutil.copytree("rlkit", os.path.join(self.log_dir, "save_files", "rlkit"))
        shutil.copyfile(launch_file_name, os.path.join(self.log_dir, "save_files", launch_file_name))

        print("dims", dims)
        self.o_dim = dims["obs_dim"]
        self.a_dim = dims["action_dim"]
        self.r_dim = dims["reward_dim"]
        self.l_dim = dims["latent_dim"]

        # self.writer = SummaryWriter(log_dir=self.log_dir)
        self.algorithm = algorithm

        self.use_auto_entropy = use_auto_entropy
        self.alpha_net_weight_decay = alpha_net_weight_decay
        self.target_entropy = -dims["action_dim"]


        self.sampler = InPlacePathSampler(
            env=env,
            policy=agent,
            max_path_length=self.max_path_length,
        )
        self.replay_buffer = MultiTaskReplayBuffer(
            self.replay_buffer_size,
            env,
            self.train_tasks,
        )

        self.prior_enc_replay_buffer = MultiTaskReplayBuffer(
            self.replay_buffer_size,
            env,
            self.train_tasks,
        )

        self.online_enc_replay_buffer = MultiTaskReplayBuffer(
            self.num_steps_prior,
            env,
            self.train_tasks,
        )

        self.bisim_target_sample_buffer_size = 100000 * len(self.train_tasks)
        self.bisim_target_sample_buffer = MultiTaskReplayBuffer(
            self.bisim_target_sample_buffer_size + 1000,
            env,
            self.train_tasks,
        )

        if self.use_episodic_online_buffer:
            self.episodic_online_buffer = EpisodicOnlineReplayBuffer(num_task=len(self.train_tasks), max_episode_per_buffer=10)


        self._n_env_steps_total = 0
        self._init_n_env_steps = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []

    def get_task_info(self, indices):
        label = []

        if self.env_name in ["cheetah-vel-inter", "cheetah-vel-extra"]:
            for idx in indices:
                label.append(round(self.total_tasks_dict_list[idx]['velocity'], 4))
            indices, label = np.array(indices), np.array(label)

            sorted_indices = label.argsort()
            indices = indices[sorted_indices].tolist()
            label = label[sorted_indices].tolist()

        elif self.env_name in ["ant-goal-inter", "ant-goal-extra", "ant-goal-extra-hard"]:
            for idx in indices:
                label.append(np.around(self.total_tasks_dict_list[idx]['goal'], 4))
            indices, label = np.array(indices), np.array(label)

            goal_dists = []
            for i in range(len(label)):
                goal_dists.append(np.sqrt(label[i][0] ** 2 + label[i][1] ** 2))
            goal_dists = np.array(goal_dists)
            sorted_indices = goal_dists.argsort()
            indices = indices[sorted_indices].tolist()
            label = label[sorted_indices].tolist()

        elif self.env_name in ["ant-dir-4개", "ant-dir-2개"]:
            for idx in indices:
                label.append(np.around(self.total_tasks_dict_list[idx]['goal'], 4))
            indices, label = np.array(indices), np.array(label)

            sorted_indices = label.argsort()
            indices = indices[sorted_indices].tolist()
            label = label[sorted_indices].tolist()

        elif "mass" in self.env_name or "params" in self.env_name:

            for idx in indices:
                label.append(np.around(self.total_tasks_dict_list[idx], 4))
            indices, label = np.array(indices), np.array(label)

            sorted_indices = label.argsort()
            indices = indices[sorted_indices].tolist()
            label = label[sorted_indices].tolist()

        return indices, label

    def make_exploration_policy(self, policy):
        return policy

    def make_eval_policy(self, policy):
        return policy

    def sample_task(self, is_eval=False):
        '''
        sample task randomly
        '''
        if is_eval:
            idx = np.random.randint(len(self.eval_tasks))
        else:
            idx = np.random.randint(len(self.train_tasks))
        return idx

    def train(self):
        '''
        meta-training loop
        '''
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        gt.reset()
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()

        for it_ in gt.timed_for(
                range(self.num_iterations),
                save_itrs=True,
        ):
            self._start_epoch(it_)
            self.training_mode(True)

            if it_ == 0:
                print('collecting initial pool of data for train and eval... ')
                accum_context_during_prior_steps = True if self.use_episodic_online_buffer else False
                for idx in self.train_tasks:
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    self.collect_data(self.num_initial_steps,
                                      1,
                                      np.inf,
                                      accum_context=accum_context_during_prior_steps,
                                      add_to_online_enc_buffer=True)  #
                    if self.use_episodic_online_buffer:
                        self.episodic_online_buffer.add_context(task_index=idx, context=self.agent.context)

            self._init_n_env_steps = self._n_env_steps_total

            if it_ == 0:
                print('pre training... ')
                for pretrain_step in range(self.pretrain_steps):
                    self.pretrain(pretrain_step)
                print('done for pretraining')


            if it_ == 0:
                self.bisim_target_sample_buffer.task_buffers[0].clear()

            accum_context_during_prior_steps = True if self.use_episodic_online_buffer else False
            for i in range(self.num_tasks_sample):
                idx = np.random.randint(len(self.train_tasks))
                self.task_idx = idx
                self.env.reset_task(idx)
                if self.clear_enc_buffer:
                    self.prior_enc_replay_buffer.task_buffers[idx].clear()
                    self.online_enc_replay_buffer.task_buffers[idx].clear()

                if self.num_steps_prior > 0:
                    self.collect_data(self.num_steps_prior, 1, np.inf, accum_context=accum_context_during_prior_steps)  # add_to_prior_enc_buffer=True, add_to_online_enc_buffer=True
                if self.use_episodic_online_buffer:
                    self.episodic_online_buffer.add_context(task_index=idx, context=self.agent.context)  # ([1, 497, 36])


                if self.num_steps_posterior > 0:
                    self.collect_data(self.num_steps_posterior, 1, self.update_post_train)

                if self.num_extra_rl_steps_posterior > 0:
                    self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train,
                                      add_to_prior_enc_buffer=self.make_prior_to_rl, add_to_online_enc_buffer=False,
                                      add_to_bisim_target_sample_buffer=False)
            self.agent.c_vae_curr_c = []

            sac_loss_list = [[] for i in range(26)]
            meta_train_losses = [[] for i in range(24)]

            t1 = time.time()
            for train_step in range(self.k_model):
                meta_train_loss = self.meta_train(train_step, ep=it_)
                for idx, loss in enumerate(meta_train_loss):
                    if not (loss == 0.0):
                        meta_train_losses[idx].append(loss)
            t2 = time.time()
            print("representation_time={} (s)".format(t2 - t1))

            for train_step in range(self.k_rl):
                indices = np.random.choice(self.train_tasks, self.n_meta)
                sac_losses = self._do_training(indices, train_step)
                for idx, loss in enumerate(sac_losses):
                    if loss is not None:
                        sac_loss_list[idx].append(loss)
                self._n_train_steps_total += 1
            t3 = time.time()
            print("rl_time={} (s)".format(t3 - t2))

            gt.stamp('train')

            self.training_mode(False)




            self._try_to_eval(it_)
            gt.stamp('eval')

            self._end_epoch()


    def collect_data(self,
                     num_samples,
                     resample_z_rate,
                     update_posterior_rate,
                     accum_context=False,
                     add_to_rl_buffer=True,
                     add_to_prior_enc_buffer=True,
                     add_to_online_enc_buffer=True,
                     add_to_bisim_target_sample_buffer=True,
                     epoch=0):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        if epoch == 0:
            random_task_ctxt_batch = None
        else:
            random_task_index = np.random.choice(self.train_tasks, 1, replace=False)
            random_task_ctxt_batch = self.sample_context(random_task_index, which_buffer=self.offpol_ctxt_sampling_buffer)
        self.agent.c_vae_curr_c = []
        self.agent.clear_z(random_task_ctxt_batch)

        num_transitions = 0
        if self.use_c_vae:
            sample_dirichlet_c_for_exploration_ = False
        else:
            sample_dirichlet_c_for_exploration_ = True
        while num_transitions < num_samples:
            paths, n_samples = self.sampler.obtain_samples(max_samples=num_samples - num_transitions,
                                                           max_trajs=update_posterior_rate,
                                                           accum_context=accum_context,
                                                           resample=resample_z_rate,
                                                           sample_dirichlet_c_for_exploration=sample_dirichlet_c_for_exploration_)  # self.max_path_length

            num_transitions += n_samples

            if add_to_rl_buffer:
                self.replay_buffer.add_paths(self.task_idx, paths)

            if add_to_prior_enc_buffer and self.prior_enc_replay_buffer.task_buffers[self.task_idx].size() < self.prior_enc_buffer_size:
                self.prior_enc_replay_buffer.add_paths(self.task_idx, paths)

            if add_to_online_enc_buffer:
                self.online_enc_replay_buffer.add_paths(self.task_idx, paths)


            if self.use_first_samples_for_bisim_samples:
                if self.bisim_target_sample_buffer.task_buffers[0].size() < self.bisim_target_sample_buffer_size:
                    self.bisim_target_sample_buffer.add_paths(0, paths)
            else:
                self.bisim_target_sample_buffer.add_paths(0, paths)


            if update_posterior_rate != np.inf:
                context = self.sample_context(self.task_idx, which_buffer="online")
                self.agent.infer_posterior(context, which_enc='psi')
                sample_dirichlet_c_for_exploration_ = False

        self._n_env_steps_total = self._n_env_steps_total + num_transitions
        gt.stamp('sample')

    def _try_to_eval(self, epoch):
        logger.save_extra_data(self.get_extra_data_to_save(epoch))
        if self._can_evaluate():
            self.evaluate(epoch)

            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            table_keys = logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys

            logger.record_tabular(
                "Number of train steps total",
                self._n_train_steps_total,
            )
            logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            logger.record_tabular(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs['train'][-1]
            sample_time = times_itrs['sample'][-1]
            eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + eval_time
            total_time = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        return True

    def _can_train(self):
        return all([self.replay_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_tasks])

    def _get_action_and_info(self, agent, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        agent.set_num_steps_total(self._n_env_steps_total)
        return agent.get_action(observation, )

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    ##### Snapshotting utils #####
    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    def collect_paths(self, idx, epoch, run, return_z=False):
        self.task_idx = idx
        self.env.reset_task(idx)
        # print("task_idx :", idx)

        if epoch == 0:
            random_task_ctxt_batch = None
        else:
            random_task_index = np.random.choice(self.train_tasks, 1, replace=False)
            random_task_ctxt_batch = self.sample_context(random_task_index, which_buffer='online')
        self.agent.c_vae_curr_c = []
        self.agent.clear_z(random_task_ctxt_batch)

        paths = []
        task_z = []
        num_transitions = 0
        num_trajs = 0
        deterministic_ = False
        if self.use_c_vae:
            sample_dirichlet_c_for_exploration_ = False
        else:
            sample_dirichlet_c_for_exploration_ = True
        while num_transitions < self.num_steps_per_eval:
            path, num = self.sampler.obtain_samples(deterministic=deterministic_,
                                                    max_samples=self.num_steps_per_eval - num_transitions, max_trajs=1,
                                                    accum_context=True,
                                                    sample_dirichlet_c_for_exploration=sample_dirichlet_c_for_exploration_)

            paths += path

            num_transitions += num
            num_trajs += 1

            if num_trajs >= self.num_exp_traj_eval:  # 4
                deterministic_ = True
                sample_dirichlet_c_for_exploration_ = False
                c = self.agent.get_context_embedding(self.agent.context, which_enc='psi').detach()
                task_z.append(c)

            else:
                if epoch == 0:
                    random_task_ctxt_batch = None
                else:
                    random_task_index = np.random.choice(self.train_tasks, 1, replace=False)
                    random_task_ctxt_batch = self.sample_context(random_task_index, which_buffer='online')
                self.agent.clear_z(random_task_ctxt_batch, context_clear=False)

        self.agent.c_vae_curr_c = []

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal  # goal

        if self.dump_eval_paths:
            logger.save_extra_data(paths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))

        if return_z:
            return paths, task_z[0]  #
        else:
            return paths

    def _do_eval(self, indices, epoch):
        final_returns = []
        online_returns = []
        for idx in indices:
            all_rets = []
            for r in range(self.num_evals):
                paths = self.collect_paths(idx, epoch, r, return_z=False)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
            final_returns.append(np.mean([a[-1] for a in all_rets]))
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0)
            online_returns.append(all_rets)
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        return final_returns, online_returns

    def get_distance_matrix(self, z_lst):
        length = len(z_lst)
        m = np.zeros((length, length))
        for i in range(length):
            for j in range(length):
                diff = z_lst[i] - z_lst[j]
                euclidian = np.linalg.norm(diff, ord=2)
                m[i, j] = euclidian
        return m



    def evaluate(self, epoch):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        if self.dump_eval_paths:
            self.agent.clear_z()
            prior_paths, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                         max_samples=self.max_path_length * 20,
                                                         accum_context=False,
                                                         resample=1)
            logger.save_extra_data(prior_paths, path='eval_trajectories/prior-epoch{}'.format(epoch))

        indices = np.random.choice(self.train_tasks, len(self.eval_tasks))
        eval_util.dprint('evaluating on {} train tasks'.format(len(indices)))
        train_returns = []
        for idx in indices:
            self.task_idx = idx
            self.env.reset_task(idx)
            paths = []
            num_episode = self.num_steps_per_eval // self.max_path_length  # 3
            deterministic_ = False
            for i in range(num_episode):  #
                if i == num_episode - 1:  #
                    deterministic_ = True
                context = self.sample_context(idx)
                self.agent.infer_posterior(context, which_enc='psi')
                p, _ = self.sampler.obtain_samples(deterministic=deterministic_,
                                                   max_samples=self.max_path_length,
                                                   accum_context=False,
                                                   max_trajs=1,
                                                   resample=np.inf,
                                                   sample_dirichlet_c_for_exploration=False)
                paths += p

            if self.sparse_rewards:
                for p in paths:
                    sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                    p['rewards'] = sparse_rewards

            train_returns.append(eval_util.get_average_returns(paths))
        train_returns = np.mean(train_returns)

        print("indices _do_eval", indices)
        train_final_returns, train_online_returns = self._do_eval(indices, epoch)
        eval_util.dprint('train online returns')
        eval_util.dprint(train_online_returns)

        eval_util.dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
        test_final_returns, test_online_returns = self._do_eval(self.eval_tasks, epoch)
        eval_util.dprint('test online returns')
        eval_util.dprint(test_online_returns)

        if len(self.indistribution_tasks) > 0:
            eval_util.dprint('evaluating on {} indistribution tasks'.format(len(self.indistribution_tasks)))
            indistribution_final_returns, indistribution_online_returns = self._do_eval(self.indistribution_tasks, epoch)
            eval_util.dprint('indistribution online returns')
            eval_util.dprint(indistribution_online_returns)
        else:
            indistribution_final_returns, indistribution_online_returns = [0.0], [0.0]

        num_task = len(self.train_tasks)
        rl_buffer_size = np.zeros(num_task)
        prior_enc_buffer_size = np.zeros(num_task)
        online_enc_buffer_size = np.zeros(num_task)
        for task_idx in range(num_task):
            rl_buffer_size[task_idx] = self.replay_buffer.task_buffers[task_idx].size()
            prior_enc_buffer_size[task_idx] = self.prior_enc_replay_buffer.task_buffers[task_idx].size()
            online_enc_buffer_size[task_idx] = self.online_enc_replay_buffer.task_buffers[task_idx].size()
        print("rl_buffer_size", rl_buffer_size)
        print("prior_enc_buffer_size", prior_enc_buffer_size)
        print("online_enc_buffer_size", online_enc_buffer_size)


        print("train_final_returns", train_final_returns)
        print("test_final_returns", test_final_returns)

        train_returns = np.mean(train_returns)

        train_avg_return = np.mean(train_final_returns)
        test_avg_return = np.mean(test_final_returns)
        indistribution_avg_return = np.mean(indistribution_final_returns)

        train_avg_online_return = np.mean(train_online_returns)
        test_avg_online_return = np.mean(test_online_returns)
        indistribution_avg_online_return = np.mean(indistribution_online_returns)



        self.agent.log_diagnostics(self.eval_statistics)

        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(paths, prefix=None)

        avg_train_return = np.mean(train_final_returns)
        avg_test_return = np.mean(test_final_returns)
        avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
        avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
        # self.eval_statistics['AverageTrainReturn_all_train_tasks'] = train_returns
        # self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
        # self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return

        # Average Return log
        if "dir-2" in self.env_name:
            self.eval_statistics['test_avg_return'] = (test_avg_return * 2 + indistribution_avg_return) / 3
            self.eval_statistics['indistribution_avg_return'] = train_avg_return
            self.eval_statistics['train_avg_return'] = train_avg_return
            print("train_avg_return", train_avg_return)
            print("indistribution_avg_return", train_avg_return)
            print("test_avg_return", (test_avg_return * 2 + indistribution_avg_return) / 3)
        else:
            self.eval_statistics['test_avg_return'] = test_avg_return
            self.eval_statistics['indistribution_avg_return'] = indistribution_avg_return
            self.eval_statistics['train_avg_return'] = train_avg_return
            print("train_avg_return", train_avg_return)
            print("indistribution_avg_return", indistribution_avg_return)
            print("test_avg_return", test_avg_return)

        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

        if self.render_eval_paths:
            self.env.render_paths(paths)

        if self.plotter:
            self.plotter.draw()

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        pass

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def _do_training(self, indices):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass

    @abc.abstractmethod
    def pretrain(self, step, pretrain):
        pass


