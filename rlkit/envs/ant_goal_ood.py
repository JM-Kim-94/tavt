"""MuJoCo150버전"""
import numpy as np
import random
# from rlkit.envs.ant import AntEnv
from gym.envs.mujoco import AntEnv as AntEnv

from . import register_env


@register_env('ant-goal-ood')
class AntGoalEnv(AntEnv):
    def __init__(self, num_train_tasks=2, eval_tasks_list=[], indistribution_train_tasks_list=[], TSNE_tasks_list=[],
                        use_cfrc=True, ood="inter", use_ref_task=0):
        self._task = {}
        self.tasks = self.sample_tasks(num_train_tasks, eval_tasks_list, indistribution_train_tasks_list, TSNE_tasks_list, 
                                        use_cfrc, ood, use_ref_task)
        print("all tasks : ", self.tasks)
        self._goal = self.tasks[0]['goal']
        self.use_cfrc = use_cfrc

        super(AntGoalEnv, self).__init__()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self._goal))

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
        )

    
    def get_obs_dim(self):
        return int(np.prod(self._get_obs().shape))

    def sample_tasks(self, num_train_tasks, eval_tasks_list, indistribution_train_tasks_list, TSNE_tasks_list, 
                            use_cfrc, ood, use_ref_task):

        if ood == "inter":
            goal_train = []
            for i in range(num_train_tasks):
                prob = random.random()
                if prob < 4.0 / 15.0:
                    r = random.random() ** 0.5
                else:
                    r = (random.random() * 2.75 + 6.25) ** 0.5
                theta = random.random() * 2 * np.pi
                goal_train.append([r * np.cos(theta), r * np.sin(theta)])


        elif ood == "extra":
            goal_train = []
            for i in range(num_train_tasks):
                r = (random.random() * 5.25 + 1.0) ** 0.5
                theta = random.random() * 2 * np.pi
                goal_train.append([r * np.cos(theta), r * np.sin(theta)])
        
        elif ood == "extra-hard":
            goal_train = []
            for i in range(num_train_tasks):
                r = (random.random() * 1.75 + 2.25) ** 0.5
                theta = random.random() * 2 * np.pi
                goal_train.append([r * np.cos(theta), r * np.sin(theta)])
        

        goal_test = eval_tasks_list

        goal_indistribution = indistribution_train_tasks_list

        goal_tsne = TSNE_tasks_list

        goals = goal_train + goal_test + goal_indistribution + goal_tsne
        goals = np.array(goals)

        tasks = [{'goal': goal} for goal in goals]

        return tasks

    def _get_obs(self):
        if self.use_cfrc:
            o = np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ])
        else:
            o = np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])
        return o


    def get_all_task_idx(self):
        # return range(len(self.tasks))
        return list(range(len(self.tasks))), self.tasks

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal']  # assume parameterization of task by single vector
        self.reset()


