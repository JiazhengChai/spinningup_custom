import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os

TARGET_ENERGY=3
class VerticalMvtPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self,type='vertical_mvt_pendulum.xml',path=None,target_energy=TARGET_ENERGY):
        self.target_energy=target_energy
        self.before_init=True
        if path:
            mujoco_env.MujocoEnv.__init__(self, os.path.join(path,type), 2)
        else:
            try:
                mujoco_env.MujocoEnv.__init__(self, 'vertical_mvt_pendulum.xml', 2)
            except:
                print('Error.')
                print('Please specify the folder in which the '+type+ ' can be found.')
                exit()
        utils.EzPickle.__init__(self)
        self.before_init =False

    def step(self, action):
        z_before=self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        z_after = self.sim.data.qpos[0]
        ob = self._get_obs()

        a_energy_penalty=np.squeeze(np.abs(z_after-z_before)*np.abs(action))

        stick_energy=np.abs(0.0417*ob[4])*ob[4]+4.905*(1-ob[2])

        alive_bonus = 10

        r= -(self.target_energy-stick_energy)**2  +alive_bonus - 0.01 * ob[0]** 2 - 0.01*a_energy_penalty
        if self.before_init:
            done = bool(ob[2] < 0)
        else:
            done = bool(ob[2] < 0 or np.abs(ob[4])<0.0001)
        #done=False
        return ob, r, done, {'actuator penalty':a_energy_penalty,'stick energy':stick_energy,'delta target energy':np.abs(self.target_energy-stick_energy)}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos[:1],  # cart x pos
            np.sin(self.sim.data.qpos[1:]),  # link angles
            np.cos(self.sim.data.qpos[1:]),
            np.clip(self.sim.data.qvel, -10, 10),
            np.clip(self.sim.data.qfrc_constraint, -10, 10)
        ]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.randn(self.model.nv) * .1
        )
        return self._get_obs()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.3
        v.cam.lookat[0] = 0
        v.cam.lookat[1] = -1
        v.cam.lookat[2] = 0
        v.cam.elevation = 0






