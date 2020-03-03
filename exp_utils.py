# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse


def compose_name(model, wiki, env, prefix):
    wiki_name = '{}wiki'.format(wiki)
    env_name = env.split(':')[-1].split('-')[0]
    return '{}:{}:{}:{}'.format(env_name, model, wiki_name, prefix)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Scalable Agent', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env', type=str, default='rtfm:rock_paper_scissors-v0',
                        help='Gym environment.')
    parser.add_argument('--mode', default='train',
                        choices=['train', 'test', 'test_render'],
                        help='Training or test mode.')
    parser.add_argument('--xpid', default=None,
                        help='Experiment id. Autopopulated if not filled.')
    parser.add_argument('--prefix', default='default', help='xp identifier')
    parser.add_argument('--model', default='paper_film2_tied', help='model to use')
    parser.add_argument('--resume', help='continue from a checkpoint')
    parser.add_argument('--resume_scheduler', action='store_true', help='continue from a checkpoint')
    parser.add_argument('--resume_strategy', help='how to resume', default='all', choices=['emb', 'all'])
    parser.add_argument('--shuffle_wiki', action='store_true', help='shuffle facts in the wiki')
    parser.add_argument('--wiki', default='yes', choices=['yes', 'no'], help='use wiki or not')

    # Environment settings.
    parser.add_argument('--height', type=int, default=6, help='room height')
    parser.add_argument('--width', type=int, default=6, help='room width')
    parser.add_argument('--partial_observability', action='store_true', help='enable partial observability')
    parser.add_argument('--max_placement', type=int, default=1, help='max number of agents observed per cell')
    parser.add_argument('--max_name', type=int, default=8, help='entity-centric description length')
    parser.add_argument('--max_inv', type=int, default=8, help='inventory length')
    parser.add_argument('--max_wiki', type=int, default=80, help='wiki description length')
    parser.add_argument('--max_task', type=int, default=40, help='task description length')
    parser.add_argument('--time_penalty', type=float, default=-0.02, help='per step time penalty')
    parser.add_argument('--pretrained_emb', action='store_true', help='use pretrained embeddings')
    parser.add_argument('--demb', type=int, default=30, help='task description length')
    parser.add_argument('--drnn', type=int, default=100, help='task description length')
    parser.add_argument('--drnn_small', type=int, default=10, help='task description length')
    parser.add_argument('--drep', type=int, default=400, help='task description length')

    # Training settings.
    parser.add_argument('--disable_checkpoint', action='store_true',
                        help='Disable saving checkpoint.')
    parser.add_argument('--savedir', default='checkpoints',
                        help='Root dir where experiment data will be saved.')
    parser.add_argument('--num_actors', default=30, type=int, metavar='N',
                        help='Number of actors.')
    parser.add_argument('--total_frames', default=int(1e8), type=int, metavar='T',
                        help='Total environment frames to train for.')
    parser.add_argument('--batch_size', default=24, type=int, metavar='B',
                        help='Learner batch size.')
    parser.add_argument('--unroll_length', default=80, type=int, metavar='T',
                        help='The unroll length (time dimension; default: 64).')
    parser.add_argument('--queue_timeout', default=1, type=int,
                        metavar='S', help='Error timeout for queue.')
    # parser.add_argument('--num_buffers', default=40, type=int,
    #                     metavar='N', help='Number of shared-memory buffers.')
    parser.add_argument('--num_threads', default=4, type=int,
                        metavar='N', help='Number learner threads.')
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA.')

    # Loss settings.
    parser.add_argument('--entropy_cost', default=0.0006, type=float,
                        help='Entropy cost/multiplier.')
    parser.add_argument('--baseline_cost', default=0.5, type=float,
                        help='Baseline cost/multiplier.')
    parser.add_argument('--discounting', default=0.99, type=float,
                        help='Discounting factor.')
    parser.add_argument('--reward_clipping', default='abs_one',
                        choices=['abs_one', 'soft_asymmetric', 'none'],
                        help='Reward clipping.')

    # Optimizer settings.
    parser.add_argument('--learning_rate', default=0.0005, type=float,
                        metavar='LR', help='Learning rate.')
    parser.add_argument('--alpha', default=0.99, type=float,
                        help='RMSProp smoothing constant.')
    parser.add_argument('--momentum', default=0, type=float,
                        help='RMSProp momentum.')
    parser.add_argument('--epsilon', default=0.01, type=float,
                        help='RMSProp epsilon.')

    # Additional Flags.
    parser.add_argument('--random_agent', action='store_true',
                        help='Use a random agent to test the env.')
    return parser
