# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
import numpy as np

class EnvSpec(object):
    def __init__(self, obs_dim, act_dim, horizon):
        self.observation_dim = obs_dim
        self.action_dim = act_dim
        self.horizon = horizon

class GymEnv(object):
    def __init__(self, env_name):
        env = gym.make(env_name)
        env.reset()
        self.env = env
        self.env_id = env.spec.id

        self._horizon = env.spec.max_episode_steps
        print(f"Horizon: {self._horizon}")

        try:
            self._action_dim = self.env.env.action_dim
        except AttributeError:
            self._action_dim = self.env.env.action_space.shape[0]

        self._observation_dim = self.env.env.observation_space.shape[0]

        # Specs
        self.spec = EnvSpec(self._observation_dim, self._action_dim, self._horizon)

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def observation_dim(self):
        return self._observation_dim

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def visualize_policy(self, policy, horizon=1000, num_episodes=1, mode='exploration'):
        self.env.env.visualize_policy(policy, horizon, num_episodes, mode)

    def evaluate_policy(self, policy,
                        num_episodes=5,
                        horizon=None,
                        gamma=1,
                        visual=False,
                        percentile=[],
                        get_full_dist=False,
                        mean_action=False,
                        init_state=None,
                        terminate_at_done=True,
                        save_video_location=None,
                        seed=None):

        if seed is not None:
            self.env._seed(seed)
        horizon = self._horizon if horizon is None else horizon
        mean_eval, std, min_eval, max_eval = 0.0, 0.0, -1e8, -1e8
        ep_returns = np.zeros(num_episodes)

        if save_video_location != None:
            self.env.monitor.start(save_video_location, force=True)

        for ep in range(num_episodes):

            if init_state is not None:
                o = self.reset()
                self.env.set_state(init_state[0], init_state[1])
                o = self.env._get_obs()
            else:
                o = self.reset()

            t, done = 0, False
            while t < horizon and (done == False or terminate_at_done == False):
                if visual == True:
                    self.render()
                if mean_action:
                    a = policy.get_action(o)[1]['mean']
                else:
                    a = policy.get_action(o)[0]
                o, r, done, _ = self.step(a)
                ep_returns[ep] += (gamma ** t) * r
                t += 1

        if save_video_location != None:
            self.env.monitor.close()

        mean_eval, std = np.mean(ep_returns), np.std(ep_returns)
        min_eval, max_eval = np.amin(ep_returns), np.amax(ep_returns)
        base_stats = [mean_eval, std, min_eval, max_eval]

        percentile_stats = []
        full_dist = []

        for p in percentile:
            percentile_stats.append(np.percentile(ep_returns, p))

        if get_full_dist == True:
            full_dist = ep_returns

        return [base_stats, percentile_stats, full_dist]
