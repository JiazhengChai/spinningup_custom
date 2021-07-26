import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os

DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}

class FullCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 path=None,
                 xml_file='full_cheetah_heavyv3.xml',
                 walkstyle='',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0.1,
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 energy_weights=0.,
                 speed=0):
        utils.EzPickle.__init__(**locals())
        self.speed = speed
        self._forward_reward_weight = forward_reward_weight
        self.walkstyle=walkstyle

        self.joint_list=['bthighL','bshinL','bfootL','fthighL','fshinL','ffootL',
                         'bthighR', 'bshinR', 'bfootR', 'fthighR', 'fshinR', 'ffootR']

        self._ctrl_cost_weight = ctrl_cost_weight
        self.energy_weights=energy_weights
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        if path is not None:
            mujoco_env.MujocoEnv.__init__(self, os.path.join(path, xml_file), 5)
        else:
            mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        if len(self.walkstyle)>0:
            if self.walkstyle == "trot":
                action[6:9] = -1 * action[0:3]  # bR equals negative bL
                action[9:12] = -1 * action[3:6]  # fR equals negative fL

            elif self.walkstyle == "gallop":
                action[6:12] = action[0:6]

        states_angle = []
        for j in self.joint_list:
            states_angle.append(self.sim.data.get_joint_qpos(j))

        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]

        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        if self.speed != 0:
            delta_speed=-abs(x_velocity-self.speed) #negative as we want to minimize this delta
            x_velocity=delta_speed

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        next_states_angle = []
        for j in self.joint_list:
            next_states_angle.append(self.sim.data.get_joint_qpos(j))

        reward = forward_reward - ctrl_cost
        done = False

        energy = 0
        for i in range(len(self.joint_list)):
            delta_theta = np.abs(next_states_angle[i] - states_angle[i])
            energy = energy + np.abs(action[i]) * delta_theta

        reward -= self.energy_weights*energy
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'energy'    : energy,
            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'ori_reward':forward_reward-ctrl_cost
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
