# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The environment class."""

import torch


def _format_frame(frame):
    frame = torch.from_numpy(frame)
    return frame.view((1, 1) + frame.shape)  # (...) -> (T,B,...).


class Environment:

    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)

        out = dict(
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
        )

        frame = self.gym_env.reset()
        if isinstance(frame, dict):
            out.update({k: v.view((1, 1) + v.shape) for k, v in frame.items()})
        else:
            out['frame'] = _format_frame(frame)
        return out

    def step(self, action):
        frame, reward, done, unused_info = self.gym_env.step(action.item())
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if done:
            frame = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)

        out = dict(
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step
        )
        if isinstance(frame, dict):
            out.update({k: v.view((1, 1) + v.shape) for k, v in frame.items()})
        else:
            out['frame'] = _format_frame(frame)
        return out

    def close(self):
        self.gym_env.close()
