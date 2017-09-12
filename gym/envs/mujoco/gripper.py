import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class GripperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'gripper.xml', 2)

    def _step(self, a):
        #Get the position of Target and our
        pos_source = self.get_body_com("source_spot")
        pos_spot = self.get_body_com("target_spot")

        #Calculate distance
        dist = np.linalg.norm(pos_source - pos_spot)
        reward_dist = - dist #Reward based on distance

        #Small movements are rewarded more
        reward_ctrl = - np.square(a).sum()

        #total reward
        reward = reward_dist + reward_ctrl

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        #You can run inifinite movements
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    #Change viewer to see in the upward position
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance *= 1.5
        self.viewer.cam.elevation = -90

    #Reset model to a "random" location
    def reset_model(self):
        #Choose a random initial position
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        
        #Choose a random position for source and target
        self.target_goal = self.np_random.uniform(low=-.32, high=.32, size=2)
        self.source_goal = self.np_random.uniform(low=-.22, high=.22, size=2)
        self.goal = np.concatenate((self.source_goal, self.target_goal))
        #Set position for target
        qpos[-4:] = self.goal
        qvel[-4:] = 0

        #set the new "state"
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:5]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[-4:],
            self.model.data.qvel.flat[:5],
            self.get_body_com("source_spot") - self.get_body_com("target_spot")
        ])
