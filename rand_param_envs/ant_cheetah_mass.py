

# import numpy as np
# import random
# from . import register_env
# from .half_cheetah import HalfCheetahEnv

# @register_env('cheetah-mass')
# class HalfCheetahMassEnv(HalfCheetahEnv):

#     def __init__(self, task={}, num_train_tasks=10, eval_tasks_list=1, indistribution_train_tasks_list=1, TSNE_tasks_list=2, ood_type='inter'):
#         super(HalfCheetahMassEnv, self).__init__()

#         self.log_scale_limit = 3.0
#         self.save_parameters()

#         self._task = task
#         self.tasks = self.sample_tasks(num_train_tasks, eval_tasks_list, indistribution_train_tasks_list, TSNE_tasks_list, ood_type)
#         print("all tasks : ", len(self.tasks))

#         self.reset_task(0)

#     # def step(self, action):
#     #     xposbefore = self.sim.data.qpos[0]
#     #     self.do_simulation(action, self.frame_skip)
#     #     xposafter = self.sim.data.qpos[0]
#     #
#     #     forward_vel = (xposafter - xposbefore) / self.dt
#     #     forward_reward = -1.0 * abs(forward_vel - self._goal_vel)
#     #     ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))
#     #
#     #     observation = self._get_obs()
#     #     reward = forward_reward - ctrl_cost
#     #     done = False
#     #     infos = dict(reward_forward=forward_reward, reward_ctrl=-ctrl_cost, task=self._task)
#     #
#     #     return observation, reward, done, infos
    
#     def step(self, action):
#         xposbefore = self.sim.data.qpos[0]
#         self.do_simulation(action, self.frame_skip)
#         xposafter = self.sim.data.qpos[0]

#         ob = self._get_obs()
#         reward_ctrl = -0.1 * np.square(action).sum()
#         reward_run = (xposafter - xposbefore) / self.dt
#         reward = reward_ctrl + reward_run
#         terminated = False

#         if self.render_mode == "human":
#             self.render()
#         return (
#             ob,
#             reward,
#             terminated,
#             False,
#             dict(reward_run=reward_run, reward_ctrl=reward_ctrl),
#         )

#     def _get_obs(self):
#         return np.concatenate(
#             [
#                 self.sim.data.qpos.flat[1:],
#                 self.sim.data.qvel.flat,
#             ]
#         )

#     def reset_model(self):
#         qpos = self.init_qpos + self.np_random.uniform(
#             low=-0.1, high=0.1, size=self.model.nq
#         )
#         qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
#         self.set_state(qpos, qvel)
#         return self._get_obs()

#     def viewer_setup(self):
#         assert self.viewer is not None
#         self.viewer.cam.distance = self.model.stat.extent * 0.5

#     def get_obs_dim(self):
#         return int(np.prod(self._get_obs().shape))

#     def get_one_rand_params(self, task='inter', eval_mode='train', value=0):  
#         new_params = {}

#         mass_size_= np.prod(self.model.body_mass.shape)
#         if task=="inter":
#             if eval_mode=="train":
#                 prob = random.random()
#                 if prob >= 0.5:
#                     body_mass_multiplyers = random.uniform(0, 0.5)
#                 else:
#                     body_mass_multiplyers = random.uniform(3.0, 3.5)
#             elif eval_mode=="eval":
#                 body_mass_multiplyers = value

#         elif task=="extra":
#             if eval_mode=="train":
#                 body_mass_multiplyers = random.uniform(1.0, 2.5)  # 1.0, 2.5 : hard / 0.5, 3.0 : easy
#             elif eval_mode=="eval":
#                 body_mass_multiplyers = value

#         body_mass_multiplyers = np.array([body_mass_multiplyers for _ in range(mass_size_)])
#         body_mass_multiplyers = np.array(1.5) ** body_mass_multiplyers
#         body_mass_multiplyers = np.array(body_mass_multiplyers).reshape(self.model.body_mass.shape)
#         new_params['body_mass'] = self.init_params['body_mass'] * body_mass_multiplyers

#         return new_params

#     def sample_tasks(self, num_train_tasks, eval_tasks_list, indistribution_train_tasks_list, TSNE_tasks_list, ood_type):

#         train_tasks, eval_tasks, indistribution_tasks, tsne_tasks = [], [], [], []

#         # get_one_rand_params(self, task='inter', eval_mode='train', value=0):  
#         """train_task"""  # [0,0.5] + [3,3.5]
#         for _ in range(num_train_tasks):  # 
#             new_params = self.get_one_rand_params(task=ood_type, eval_mode='train')
#             train_tasks.append(new_params)
        
#         """eval_task"""  # 16 
#         for v in eval_tasks_list:  
#             new_params = self.get_one_rand_params(task=ood_type, eval_mode='eval', value=v)
#             eval_tasks.append(new_params)

#         """indistribution_task"""  
#         for v in indistribution_train_tasks_list:  # 4  ood='inter', eval_mode='train', set_rand_param="body_mass", eval_idx=0, value=0
#             new_params = self.get_one_rand_params(task=ood_type, eval_mode='eval', value=v)
#             indistribution_tasks.append(new_params)
        
#         """tsne_task for visualization"""  # 48
#         tsne_tasks = eval_tasks + indistribution_tasks
        
#         """total tasks list"""
#         param_sets = train_tasks + eval_tasks + indistribution_tasks + tsne_tasks
#         return param_sets

#     def save_parameters(self):
#         print("initial self.model.body_mass", self.model.body_mass)
#         self.init_params = {}
#         self.init_params['body_mass'] = self.model.body_mass
#         self.cur_params = self.init_params

#     def get_all_task_idx(self):
#         return list(range(len(self.tasks))), self.tasks

#     def set_bodymass_task(self, task):
#         param_val_new = task['body_mass']
#         param_val_before = getattr(self.model, "body_mass")
#         # print("param_val_before", param_val_before)
#         assert param_val_before.shape == param_val_new.shape, 'shapes of new parameter value and old one must match'
#         for i in range(len(param_val_before)):
#             self.model.body_mass[i] = param_val_new[i]
#         param_val_after = getattr(self.model, "body_mass")
#         # print("param_val_after", param_val_after)
#         self.cur_params = task

#     def reset_task(self, idx):
#         self._task = self.tasks[idx]
#         # self._goal_vel = self._task['velocity']
#         # self._goal = self._goal_vel
#         self._goal = idx
#         self.set_bodymass_task(self._task)
#         self.reset()
#         # print("self.model.body_mass", self.model.body_mass)






import numpy as np
from rand_param_envs.base2 import RandomEnv
from rand_param_envs.gym import utils

# from . import register_env

# @register_env('cheetah-mass')
class AntMassEnv(RandomEnv, utils.EzPickle):
    def __init__(self, log_scale_limit=3.0):
        RandomEnv.__init__(self, log_scale_limit, 'ant.xml')
        utils.EzPickle.__init__(self)
    
    def _step(self, action):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]

        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0

        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()

        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    
    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.distance = self.model.stat.extent * 0.5
