

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

    def __init__(self, log_scale_limit, file_name, set_other_params, rand_params=RAND_PARAMS, **kwargs):
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
        self.set_other_params = set_other_params
        print("self.set_other_params", self.set_other_params)
        self.save_parameters()
    
    # def get_one_rand_params(self, ood='inter', eval_mode='train', set_rand_param="body_mass", value=0):  
    #                             # ood='inter'/'extra'  
    #                             # eval_mode='train'/'eval'/
    #     new_params = {}

    #     if 'body_mass' in self.rand_params:
    #         # body_mass_multiplyers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=self.model.body_mass.shape)
    #         # print("size", self.model.body_mass.shape)  # (8, 1)
            

    #         size, body_mass_multiplyers = np.prod(self.model.body_mass.shape), []
    #         while len(body_mass_multiplyers) < size:
    #             temp = np.random.uniform(-self.log_scale_limit, self.log_scale_limit)
    #             if ood=="inter":
    #                 if eval_mode=="train":
    #                     if np.abs(temp) < 1 or np.abs(temp) > 2:
    #                         body_mass_multiplyers.append(np.array(1.5) ** temp)
    #                 elif eval_mode=="eval":
    #                     # if np.abs(temp) > 1 and np.abs(temp) < 2:
    #                     #     body_mass_multiplyers.append(np.array(1.5) ** temp)
    #                     if set_rand_param == 'body_mass':
    #                         body_mass_multiplyers.append(np.array(1.5) ** value)
    #                     else:
    #                         body_mass_multiplyers.append(np.array(1.5) ** 0.0)
    #         body_mass_multiplyers = np.array(body_mass_multiplyers).reshape(self.model.body_mass.shape)
    #         # print("body_mass_multiplyers", body_mass_multiplyers.shape)  # (8, 1)
    #         new_params['body_mass'] = self.init_params['body_mass'] * body_mass_multiplyers

    #     # body_inertia
    #     if 'body_inertia' in self.rand_params:
    #         # body_inertia_multiplyers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=self.model.body_inertia.shape)
    #         # print("body_inertia_multiplyers", body_inertia_multiplyers.shape)  # (8, 3)
    #         size, body_inertia_multiplyers = np.prod(self.model.body_inertia.shape), []
    #         while len(body_inertia_multiplyers) < size:
    #             temp = np.random.uniform(-self.log_scale_limit, self.log_scale_limit)
    #             if ood=="inter":
    #                 if eval_mode=="train":
    #                     if np.abs(temp) < 1 or np.abs(temp) > 2:
    #                         body_inertia_multiplyers.append(np.array(1.5) ** temp)
    #                 elif eval_mode=="eval":
    #                     # if np.abs(temp) > 1 and np.abs(temp) < 2:
    #                     #     body_inertia_multiplyers.append(np.array(1.5) ** temp)
    #                     if set_rand_param == 'body_inertia':
    #                         body_inertia_multiplyers.append(np.array(1.5) ** value)
    #                     else:
    #                         body_inertia_multiplyers.append(np.array(1.5) ** 0.0)
    #         body_inertia_multiplyers = np.array(body_inertia_multiplyers).reshape(self.model.body_inertia.shape)
    #         # print("body_inertia_multiplyers", body_inertia_multiplyers.shape)  # (8, 1)
    #         new_params['body_inertia'] = body_inertia_multiplyers * self.init_params['body_inertia']

    #     # damping -> different multiplier for different dofs/joints
    #     if 'dof_damping' in self.rand_params:
    #         # dof_damping_multipliers = np.array(1.3) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.model.dof_damping.shape)
    #         # print("dof_damping_multipliers", dof_damping_multipliers.shape)  # (9, 1)
    #         size, dof_damping_multipliers = np.prod(self.model.dof_damping.shape), []
    #         while len(dof_damping_multipliers) < size:
    #             temp = np.random.uniform(-self.log_scale_limit, self.log_scale_limit)
    #             if ood=="inter":
    #                 if eval_mode=="train":
    #                     if np.abs(temp) < 1 or np.abs(temp) > 2:
    #                         dof_damping_multipliers.append(np.array(1.3) ** temp)
    #                 elif eval_mode=="eval":
    #                     # if np.abs(temp) > 1 and np.abs(temp) < 2:
    #                     #     dof_damping_multipliers.append(np.array(1.3) ** temp)
    #                     if set_rand_param == 'dof_damping':
    #                         dof_damping_multipliers.append(np.array(1.3) ** value)
    #                     else:
    #                         dof_damping_multipliers.append(np.array(1.3) ** 0.0)
    #         dof_damping_multipliers = np.array(dof_damping_multipliers).reshape(self.model.dof_damping.shape)
    #         # print("dof_damping_multipliers", dof_damping_multipliers.shape)  # (8, 1)
    #         new_params['dof_damping'] = np.multiply(self.init_params['dof_damping'], dof_damping_multipliers)

    #     # friction at the body components
    #     if 'geom_friction' in self.rand_params:
    #         # dof_damping_multipliers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.model.geom_friction.shape)
    #         # print("dof_damping_multipliers", dof_damping_multipliers.shape)  # (8, 3)
    #         size, geom_friction_multipliers = np.prod(self.model.geom_friction.shape), []
    #         while len(geom_friction_multipliers) < size:
    #             temp = np.random.uniform(-self.log_scale_limit, self.log_scale_limit)
    #             if ood=="inter":
    #                 if eval_mode=="train":
    #                     if np.abs(temp) < 1 or np.abs(temp) > 2:
    #                         geom_friction_multipliers.append(np.array(1.5) ** temp)
    #                 elif eval_mode=="eval":
    #                     # if np.abs(temp) > 1 and np.abs(temp) < 2:
    #                     #     geom_friction_multipliers.append(np.array(1.5) ** temp)
    #                     if set_rand_param == 'geom_friction':
    #                         geom_friction_multipliers.append(np.array(1.5) ** value)
    #                     else:
    #                         geom_friction_multipliers.append(np.array(1.5) ** 0.0)
    #         geom_friction_multipliers = np.array(geom_friction_multipliers).reshape(self.model.geom_friction.shape)
    #         # print("geom_friction_multipliers", geom_friction_multipliers.shape)  # (8, 1)
    #         new_params['geom_friction'] = np.multiply(self.init_params['geom_friction'], geom_friction_multipliers)
        
    #     return new_params

    def get_one_rand_params(self, ood='inter', eval_mode='train', set_rand_param="body_mass", eval_idx=0, value=0):  
                                # ood='inter'/'extra'  
                                # eval_mode='train'/'eval'/
        new_params = {}

        if 'body_mass' in self.rand_params:
            # body_mass_multiplyers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=self.model.body_mass.shape)
            # print("size", self.model.body_mass.shape)  # (8, 1)   

            flag, size_= True, np.prod(self.model.body_mass.shape)
            while flag:
                if ood=="inter":
                    if eval_mode=="train":
                        body_mass_multiplyers = [random.uniform(-self.log_scale_limit, self.log_scale_limit) for s in range(size_)]
                        body_mass_multiplyers = np.array(body_mass_multiplyers)
                        l = abs(body_mass_multiplyers).max()
                        # if l > 2 or l < 1:  # easy
                        if l > 2.5 or l < 0.5:  # hard
                            body_mass_multiplyers = np.array(1.5) ** body_mass_multiplyers
                            flag = False
                    elif eval_mode=="eval":
                        if set_rand_param == 'body_mass':
                            # print("size_", size_)
                            # print("eval_idx", eval_idx)
                            body_mass_multiplyers = np.zeros(size_)
                            body_mass_multiplyers[eval_idx] = value
                            body_mass_multiplyers = np.array(1.5) ** body_mass_multiplyers
                            flag = False

                elif ood=="extra":
                    if eval_mode=="train":
                        body_mass_multiplyers = [random.uniform(-self.log_scale_limit, self.log_scale_limit) for s in range(size_)]
                        body_mass_multiplyers = np.array(body_mass_multiplyers)
                        l = abs(body_mass_multiplyers).max()
                        if l < 2 and l > 1:
                            body_mass_multiplyers = np.array(1.5) ** body_mass_multiplyers
                            flag = False
                    elif eval_mode=="eval":
                        if set_rand_param == 'body_mass':
                            # print("size_", size_)
                            # print("eval_idx", eval_idx)
                            body_mass_multiplyers = np.zeros(size_)
                            body_mass_multiplyers[eval_idx] = value
                            body_mass_multiplyers = np.array(1.5) ** body_mass_multiplyers
                            flag = False
            body_mass_multiplyers = np.array(body_mass_multiplyers).reshape(self.model.body_mass.shape)
            # print("body_mass_multiplyers", body_mass_multiplyers.shape)  # (8, 1)
            new_params['body_mass'] = self.init_params['body_mass'] * body_mass_multiplyers

        # body_inertia
        if 'body_inertia' in self.rand_params:
            if self.set_other_params:
                body_inertia_multiplyers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=self.model.body_inertia.shape)
                new_params['body_inertia'] = body_inertia_multiplyers * self.init_params['body_inertia']

        # damping -> different multiplier for different dofs/joints
        if 'dof_damping' in self.rand_params:
            if self.set_other_params:
                dof_damping_multipliers = np.array(1.3) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.model.dof_damping.shape)
                new_params['dof_damping'] = self.init_params['dof_damping'] * dof_damping_multipliers

        # friction at the body components
        if 'geom_friction' in self.rand_params:
            if self.set_other_params:
                geom_friction_multipliers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.model.geom_friction.shape)
                # new_params['geom_friction'] = np.multiply(self.init_params['geom_friction'], geom_friction_multipliers)
                new_params['geom_friction'] = self.init_params['geom_friction'] * geom_friction_multipliers
        
        return new_params
        

    def sample_tasks(self, n_train_tasks, n_eval_tasks, n_indistribution_tasks, n_tsne_tasks, ood='inter', target_eval_value=1.5):

        train_tasks, eval_tasks, indistribution_tasks, tsne_tasks = [], [], [], []

        """train_task"""
        for _ in range(n_train_tasks):  # 100  ood='inter', eval_mode='train', set_rand_param="body_mass", eval_idx=0, value=0
            new_params = self.get_one_rand_params(ood='inter', eval_mode='train')
            train_tasks.append(new_params)
        
        body_mass_size = np.prod(self.model.body_mass.shape)
        """eval_task"""  # 16 
        for ood_idx in range(body_mass_size):  
            new_params = self.get_one_rand_params(ood='inter', eval_mode='eval', set_rand_param="body_mass", eval_idx=ood_idx, value=target_eval_value)
            eval_tasks.append(new_params)
            new_params = self.get_one_rand_params(ood='inter', eval_mode='eval', set_rand_param="body_mass", eval_idx=ood_idx, value=-1*target_eval_value)
            eval_tasks.append(new_params)

        """indistribution_task"""  # 32
        # for ood_idx in range(body_mass_size):  
            # new_params = self.get_one_rand_params(ood='inter', eval_mode='eval', set_rand_param="body_mass", eval_idx=ood_idx, value=0.5)
            # indistribution_tasks.append(new_params)
            # new_params = self.get_one_rand_params(ood='inter', eval_mode='eval', set_rand_param="body_mass", eval_idx=ood_idx, value=-0.5)
            # indistribution_tasks.append(new_params)
            # new_params = self.get_one_rand_params(ood='inter', eval_mode='eval', set_rand_param="body_mass", eval_idx=ood_idx, value=2.75)
            # indistribution_tasks.append(new_params)
            # new_params = self.get_one_rand_params(ood='inter', eval_mode='eval', set_rand_param="body_mass", eval_idx=ood_idx, value=-2.75)
            # indistribution_tasks.append(new_params)
        for _ in range(n_indistribution_tasks):  # 4  ood='inter', eval_mode='train', set_rand_param="body_mass", eval_idx=0, value=0
            new_params = self.get_one_rand_params(ood='inter', eval_mode='train')
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
        if 'body_mass' in self.rand_params:
            self.init_params['body_mass'] = self.model.body_mass

        # body_inertia
        if 'body_inertia' in self.rand_params:
            self.init_params['body_inertia'] = self.model.body_inertia

        # damping -> different multiplier for different dofs/joints
        if 'dof_damping' in self.rand_params:
            self.init_params['dof_damping'] = self.model.dof_damping

        # friction at the body components
        if 'geom_friction' in self.rand_params:
            self.init_params['geom_friction'] = self.model.geom_friction
        self.cur_params = self.init_params