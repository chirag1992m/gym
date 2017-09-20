import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class SimpleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'simple.xml', 4)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        dist = self.get_body_com("object") - self.get_body_com("goal")

        reward_ctrl = - np.square(a).sum()
        reward_dist = - np.linalg.norm(dist)

        self.do_simulation(a, self.frame_skip)

        reward = reward_dist + reward_ctrl
        ob = self._get_obs()

        return ob, reward, False, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 10.0
        self.viewer.cam.elevation = -90

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos.flat, qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-2, high=2, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-2, high=2, size=self.model.nv)
        )
        return self._get_obs()
