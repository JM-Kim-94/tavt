



import numpy as np
import random
from gym.envs.mujoco import AntEnv as AntEnv
from . import register_env

@register_env('ant-dir-2')
class AntDirEnv(AntEnv):
    def __init__(self, num_train_tasks=2, eval_tasks_list=[], indistribution_train_tasks_list=[], TSNE_tasks_list=[],
                        ood_type='inter', done_flase=False, use_ref_task=0):
        self._task = {}
        self.tasks = self.sample_tasks(num_train_tasks, eval_tasks_list, indistribution_train_tasks_list, TSNE_tasks_list, 
                                        ood_type, done_flase, use_ref_task)
        print("all tasks : ", self.tasks)
        self._goal = self.tasks[0]['goal']
        self.done_flase = done_flase

        # xml_path = 'ant.xml'
        # super().__init__(xml_path, frame_skip=5, automatically_set_obs_and_action_space=True)
        super(AntDirEnv, self).__init__()
    
    def get_obs_dim(self):
        return int(np.prod(self._get_obs().shape))
        
    def sample_tasks(self, num_train_tasks, eval_tasks_list, indistribution_train_tasks_list, TSNE_tasks_list, 
                            ood_type, done_flase, use_ref_task):

        if ood_type == "inter":
            train_goal_dir = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]  # [i * np.pi / 2 for i in range(4)]
            eval_goal_dir  = [0.25 * np.pi, 0.75 * np.pi, 1.25 * np.pi, 1.75 * np.pi]   # [i * np.pi / 2 + np.pi / 4 for i in range(4)]
            indistribution_goal_dir = []
            tsne_goal_dir  = [0.0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi, 1.25 * np.pi, 1.5 * np.pi, 1.75 * np.pi]  # [i * np.pi / 4 for i in range(8)]  # train_goal_dir + eval_goal_dir

        elif ood_type == "extra":
            train_goal_dir = [0.0, 0.5 * np.pi]  # 2개
            eval_goal_dir  = [3 * np.pi / 4,    7 * np.pi / 4]   # 2개
            indistribution_goal_dir = [1 * np.pi / 4]  # 1개
            tsne_goal_dir  = [0.0 * np.pi / 4,
                              1.0 * np.pi / 4,
                              2.0 * np.pi / 4,
                              3.0 * np.pi / 4,
                              7.0 * np.pi / 4]  # 5개
            


        goal_dirs = train_goal_dir + eval_goal_dir + indistribution_goal_dir + tsne_goal_dir

        tasks = [{'goal': goal_dir} for goal_dir in goal_dirs]

        return tasks

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        direct = (np.cos(self._goal), np.sin(self._goal))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))

        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2]/self.dt), direct)

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum( np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                and state[2] >= 0.2 and state[2] <= 1.0

        # if self.done_flase:
        #     done = False  # not notdone
        # else:
        done = not notdone

        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )


    # def _get_obs(self):
    #     return np.concatenate([
    #         self.sim.data.qpos.flat,
    #         self.sim.data.qvel.flat,
    #         # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
    #     ])
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
        ])

    def get_all_task_idx(self):
        # return range(len(self.tasks))
        return list(range(len(self.tasks))), self.tasks

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal']  # assume parameterization of task by single vector
        self.reset()

    # def reset_model(self):
    #     qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
    #     qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()


