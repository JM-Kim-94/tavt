



import numpy as np
from rand_param_envs.hopper_rand_params import HopperRandParamsEnv

from . import register_env


@register_env('hopper-mass-ood')
class HopperRandParamsWrappedEnv(HopperRandParamsEnv):

    def __init__(self, num_train_tasks, eval_tasks_list, indistribution_train_tasks_list, TSNE_tasks_list, ood_type='inter'):

        super(HopperRandParamsWrappedEnv, self).__init__()

        self.tasks, self.tasks_value = self.sample_tasks(num_train_tasks, eval_tasks_list, 
                                        indistribution_train_tasks_list, TSNE_tasks_list, ood_type)
        self.reset_task(0)
    
    def get_obs_dim(self):
        return int(np.prod(self._get_obs().shape))

    def get_all_task_idx(self):
        # return range(len(self.tasks))
        return list(range(len(self.tasks))), self.tasks_value

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()
