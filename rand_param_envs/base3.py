
from rand_param_envs.gym.core import Env
from rand_param_envs.gym.envs.mujoco import MujocoEnv
import numpy as np
import random


class MetaEnv(Env):
    def step(self, *args, **kwargs):
        return self._step(*args, **kwargs)

    def sample_tasks(self, n_tasks):
        """
        Samples task of the meta-environment

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        raise NotImplementedError

    def set_task(self, task):
        """
        Sets the specified task to the current environment

        Args:
            task: task of the meta-learning environment
        """
        raise NotImplementedError

    def get_task(self):
        """
        Gets the task that the agent is performing in the current environment

        Returns:
            task: task of the meta-learning environment
        """
        raise NotImplementedError

    def log_diagnostics(self, paths, prefix):
        """
        Logs env-specific diagnostic information

        Args:
            paths (list) : list of all paths collected with this env during this iteration
            prefix (str) : prefix for logger
        """
        pass

class RandomEnv(MetaEnv, MujocoEnv):
    """
    This class provides functionality for randomizing the physical parameters of a mujoco model
    The following parameters are changed:
        - body_mass
        - body_inertia
        - damping coeff at the joints
    """
    RAND_PARAMS = ['body_mass', 'dof_damping', 'body_inertia', 'geom_friction']
    RAND_PARAMS_EXTENDED = RAND_PARAMS + ['geom_size']

    def __init__(self, log_scale_limit, file_name, rand_params=RAND_PARAMS, **kwargs):
        # print("log_scale_limit", log_scale_limit)  # log_scale_limit 3.0
        # print("file_name", file_name)  # file_name walker2d.xml
        # print("*args", *args)  # *args 5
        # print("rand_params", rand_params)  # rand_params ['body_mass', 'dof_damping', 'body_inertia', 'geom_friction']
        # print("**kwargs", **kwargs)  # **kwargs

        MujocoEnv.__init__(self, file_name, 4)
        assert set(rand_params) <= set(self.RAND_PARAMS_EXTENDED), \
            "rand_params must be a subset of " + str(self.RAND_PARAMS_EXTENDED)
        self.log_scale_limit = log_scale_limit            
        self.rand_params = rand_params
        self.save_parameters()
    

    def get_one_rand_params(self, task='inter', eval_mode='train', value=0):  
        new_params = {}

        mass_size_= np.prod(self.model.body_mass.shape)
        if task=="inter":
            if eval_mode=="train":  # indis = -2.5, 2.5  // test = -0.5, 0, 0.5
                prob = random.random()
                if prob >= 0.5:
                    body_mass_multiplyers = random.uniform(-3.0, -1.0)
                else:
                    body_mass_multiplyers = random.uniform(1.0, 3.0)
            elif eval_mode=="eval":
                body_mass_multiplyers = value

        # elif task=="extra":
        #     if eval_mode=="train":
        #         body_mass_multiplyers = random.uniform(1.0, 2.5)  # 1.0, 2.5 : hard / 0.5, 3.0 : easy
        #     elif eval_mode=="eval":
        #         body_mass_multiplyers = value

        body_mass_multiplyers = np.array([body_mass_multiplyers for _ in range(mass_size_)])
        body_mass_multiplyers = np.array(1.5) ** body_mass_multiplyers
        body_mass_multiplyers = np.array(body_mass_multiplyers).reshape(self.model.body_mass.shape)
        new_params['body_mass'] = self.init_params['body_mass'] * body_mass_multiplyers

        return new_params
        
    def sample_tasks(self, num_train_tasks, eval_tasks_list, indistribution_train_tasks_list, TSNE_tasks_list, ood_type):

        train_tasks, eval_tasks, indistribution_tasks, tsne_tasks = [], [], [], []

        # get_one_rand_params(self, task='inter', eval_mode='train', value=0):  
        """train_task"""  # [0,0.5] + [3,3.5]
        for _ in range(num_train_tasks):  # 
            new_params = self.get_one_rand_params(task=ood_type, eval_mode='train')
            train_tasks.append(new_params)
        
        """eval_task"""  # 16 
        for v in eval_tasks_list:  
            new_params = self.get_one_rand_params(task=ood_type, eval_mode='eval', value=v)
            eval_tasks.append(new_params)

        """indistribution_task"""  
        for v in indistribution_train_tasks_list:  # 4  ood='inter', eval_mode='train', set_rand_param="body_mass", eval_idx=0, value=0
            new_params = self.get_one_rand_params(task=ood_type, eval_mode='eval', value=v)
            indistribution_tasks.append(new_params)
        
        """tsne_task for visualization"""  # 48
        tsne_tasks = eval_tasks + indistribution_tasks
        
        """total tasks list"""
        param_sets = train_tasks + eval_tasks + indistribution_tasks + tsne_tasks
        return param_sets

    def set_task(self, task):
        for param, param_val in task.items():
            param_variable = getattr(self.model, param)
            assert param_variable.shape == param_val.shape, 'shapes of new parameter value and old one must match'
            setattr(self.model, param, param_val)
        self.cur_params = task

    def get_task(self):
        return self.cur_params

    def save_parameters(self):
        self.init_params = {}
        self.init_params['body_mass'] = self.model.body_mass

        self.cur_params = self.init_params