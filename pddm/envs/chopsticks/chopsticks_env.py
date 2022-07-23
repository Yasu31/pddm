import numpy as np
import mujoco_py
from gym import utils
from mujoco_py import load_model_from_path, MjSim

from pddm.envs import mujoco_env
from pddm.envs.robot import Robot

class ChopsticksEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    draws heavily from DClawTurnEnv
    python train.py --config ../config/chopsticks.txt --output_dir ../output
    python visualize_iteration.py --job_path ../output/chopsticks --iter_num 0
    """
    def __init__(self):
        print("creating class ChopsticksEnv")
        self.time = 0
        # this variable is used to render every once in a while?
        # but when I change this value the performance seems to change?
        # when this is set to 40 during training and to 4 in visualization it seems to create interesting movement... though I have no idea why...
        self.skip = 10

        self.n_jnt = 4
        self.n_obj = 3  # DoFs of object
        self.n_dofs = self.n_jnt + self.n_obj

        import os
        xml_path = os.path.join(os.path.dirname(__file__), '..', '..', 'env_models', 'chopsticks', 'chopsticks.xml')

        # todo maybe set pos_bounds and vel_bounds here
        self.robot = Robot(n_jnt=self.n_jnt, n_obj=self.n_obj, n_dofs=self.n_dofs, 
                           # put in arbitrary limits here for now
                           pos_bounds=[[-1, 1]]*self.n_dofs, vel_bounds=[[-1, 1]]*self.n_dofs)

        self.initializing = True
        # load mujoco model
        super().__init__(xml_path, self.skip)
        utils.EzPickle.__init__(self)
        self.initializing = False

        # set range for action
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * np.ptp(self.model.actuator_ctrlrange, axis=1)
        assert self.act_mid.shape[0] == self.act_rng.shape[0] == self.n_jnt

        assert self.n_dofs == self.sim.data.qpos.shape[0] == self.sim.data.qvel.shape[0]

    def get_reward(self, observations, actions):
        """
        Reward function for the chopsticks environment.
        Args:
            observations: (batch_size, obs_dim) or (obs_dim,)
            actions: (batch_size, act_dim) or (act_dim,)
        Returns:
            reward: (batch_size,) or (1,)
            done: (batch_size,) or (1,) True if episode is done
        """
        # reshape as needed
        if len(observations.shape) == 1:
            observations = np.expand_dims(observations, axis=0)
            batch_mode = False
        else:
            batch_mode = True

        # pos and vel, so observation is twice the number of dofs
        assert observations.shape[1] == self.n_dofs * 2
        # recover object location (joint position)
        object_qpos = observations[:, 2*self.n_jnt:2*self.n_jnt+self.n_obj]
        # second joint in object is the slide joint for z axis
        heights = object_qpos[:, 1]

        dones = [False] * observations.shape[0]
        # object higher = better
        # add bonus when height of object above 0.1
        reward = heights + 0.1 * (heights > 0.) + 0.1 * (heights > 0.05)
    
        if not batch_mode:
            return reward[0], dones[0]
        return reward, dones
        
    def get_score(self, obs):
        """
        what is this and how is it different from get_reward?
        """
        object_pos = obs[2*self.n_jnt:2*self.n_jnt+self.n_obj]
        height = object_pos[1]
        return height

    def step(self, action):
        action = np.clip(action, -1, 1)
        if not self.initializing:
            # convert value between -1 ~ 1 to action range
            action = self.act_mid + self.act_rng * action
        
        self.robot.step(self, action, step_duration=self.skip * self.model.opt.timestep)

        obs = self._get_obs()
        reward, done = self.get_reward(obs, action)
        score = self.get_score(obs)

        env_info = {"time": self.time,
                    "obs_dict": self.obs_dict,
                    "reward": reward,
                    "score": score}
        return obs, reward, done, env_info

    def _get_obs(self):
        self.robot.get_obs(self, robot_noise_ratio=0, object_noise_ratio=0)
        # cs = chopsticks
        time, cs_qp, cs_qv, obj_qp, obj_qv = self.robot.get_obs_from_cache(self, -1)
        assert len(cs_qp) == len(cs_qv) == self.n_jnt
        assert len(obj_qp) == len(obj_qv) == self.n_obj
        self.time = time

        self.obs_dict = {}
        self.obs_dict["chopstick_pos"] = cs_qp
        self.obs_dict["chopstick_vel"] = cs_qv
        self.obs_dict["object_pos"] = obj_qp
        self.obs_dict["object_vel"] = obj_qv

        return np.concatenate([cs_qp, cs_qv, obj_qp, obj_qv], dtype=np.float32)
    
    def reset_model(self):
        # these are read far away in mb_env.py and saved to the pickle file thing
        self.reset_pose = self.init_qpos
        self.reset_vel = self.init_qvel
        self.reset_goal = np.zeros(0)
    
        # no randomization for now
        
        return self.do_reset(self.reset_pose, self.reset_vel, self.reset_goal)

    def do_reset(self, reset_pose, reset_vel, reset_goal):
        print(f"reset_pose: {reset_pose}")
        self.robot.reset(self, reset_pose, reset_vel)
        self.sim.forward()
        return self._get_obs()
