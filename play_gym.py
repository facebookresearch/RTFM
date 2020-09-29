#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import argparse
import gym
import random
import getch
import rtfm.tasks
from rtfm.dynamics.monster import Player
from rtfm import featurizer as F


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment to use', default='rock_paper_scissors')
    parser.add_argument('--height', type=int, help='room size', default=12)
    parser.add_argument('--width', type=int, help='room size', default=22)
    parser.add_argument('--seed', type=int, help='seed', default=42)
    parser.add_argument('-p', '--partially_observable', action='store_true', help='only show partial observability')
    parser.add_argument('-c', '--control', action='store_true', help='assume direct control')
    parser.add_argument('-w', '--shuffle_wiki', action='store_true', help='shuffle facts in the wiki')
    args = parser.parse_args()

    featurizer = F.Concat([F.Progress(), F.ValidMoves(), F.Terminal()])
    env = gym.make('rtfm:{}-v0'.format(args.env), featurizer=featurizer, partially_observable=args.partially_observable, room_shape=(args.height, args.width), shuffle_wiki=args.shuffle_wiki)

    random.seed(args.seed)

    score = total = 0
    while True:
        env.reset()
        feat = reward = finished = won = None
        while not finished:
            if args.control:
                ch = None
                while ch not in Player.keymap:
                    print('Current score {} out of {}'.format(score, total))
                    print('Enter your command. x to quit.')
                    ch = getch.getch()
                    if ch == 'x':
                        import sys
                        sys.exit(0)
                feat, reward, finished, won = env.step(Player.keymap[ch])
            else:
                feat, reward, finished, won = env.step(random.choice(Player.valid_moves))
                time.sleep(1)
            print(feat)

            print('Reward = {}'.format(reward))
            print('Finished = {}'.format(finished))
            print('Won = {}'.format(won))
            if finished:
                score += won
                total += 1
                break
