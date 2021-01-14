# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gym
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import rnn as rnn_utils
from rtfm import featurizer as X


class Model(nn.Module):

    @classmethod
    def create_env(cls, flags, featurizer=None):
        f = featurizer or X.Concat([X.Text(), X.ValidMoves()])
        print('loading env')
        start_time = time.time()
        env = gym.make(flags.env, room_shape=(flags.height, flags.width), partially_observable=flags.partial_observability, max_placement=flags.max_placement, featurizer=f, shuffle_wiki=flags.shuffle_wiki, time_penalty=flags.time_penalty)
        print('loaded env in {} seconds'.format(time.time() - start_time))
        return env

    @classmethod
    def make(cls, flags, env):
        return cls(env.observation_space, len(env.action_space), flags.height, flags.width, env.vocab, demb=flags.demb, drnn=flags.drnn, drep=flags.drep, disable_wiki=flags.wiki == 'no')

    def __init__(self, observation_shape, num_actions, room_height, room_width, vocab, demb, drnn, drep, pretrained_emb=False, disable_wiki=False):
        super().__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.disable_wiki = disable_wiki

        self.demb = demb
        self.dconv_out = 1
        self.drep = drep
        self.drnn = drnn
        self.room_height_conv_out = room_height // 2
        self.room_width_conv_out = room_width // 2

        self.vocab = vocab
        self.emb = nn.Embedding(len(vocab), self.demb, padding_idx=vocab.word2index('pad'))

        if pretrained_emb:
            raise NotImplementedError()

        self.policy = nn.Linear(self.drep, self.num_actions)
        self.baseline = nn.Linear(self.drep, 1)

    def encode_inv(self, inputs):
        return None

    def encode_cell(self, inputs):
        return None

    def encode_wiki(self, inputs):
        return None

    def encode_task(self, inputs):
        return None

    def compute_aux_loss(self, inputs, cell, inv, wiki, task):
        T, B, *_ = task.size()
        return torch.Tensor([0] * T).to(cell.device)

    def fuse(self, inputs, cell, inv, wiki, task):
        raise NotImplementedError()

    def forward(self, inputs):
        name = inputs['name'].long()  # (T, B, H, W, placement, name_len)
        T, B, height, width, n_placement, n_text = name.size()

        # encode everything
        cell = self.encode_cell(inputs)
        inv = self.encode_inv(inputs)
        wiki = self.encode_wiki(inputs)
        task = self.encode_task(inputs)

        rep = self.fuse(inputs, cell, inv, wiki, task)

        policy_logits = self.policy(rep)
        baseline = self.baseline(rep)

        # mask out invalid actions
        action_mask = inputs['valid'].float().view(T*B, -1)
        policy_logits -= (1-action_mask) * 1e20
        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        aux_loss = self.compute_aux_loss(inputs, cell, inv, wiki, task)
        return dict(policy_logits=policy_logits, baseline=baseline, action=action, aux_loss=aux_loss)

    def run_rnn(self, rnn, x, lens):
        # embed
        emb = self.emb(x.long())
        # rnn
        packed = rnn_utils.pack_padded_sequence(emb, lengths=lens.cpu().long(), batch_first=True, enforce_sorted=False)
        packed_h, _ = rnn(packed)
        h, _ = rnn_utils.pad_packed_sequence(packed_h, batch_first=True, padding_value=0.)
        return h

    def run_selfattn(self, h, lens, scorer):
        mask = self.get_mask(lens, max_len=h.size(1)).unsqueeze(2)
        raw_scores = scorer(h)
        scores = F.softmax(raw_scores - (1-mask)*1e20, dim=1)
        context = scores.expand_as(h).mul(h).sum(dim=1)
        return context, scores

    def run_rnn_selfattn(self, rnn, x, lens, scorer):
        rnn = self.run_rnn(rnn, x, lens)
        context, scores = self.run_selfattn(rnn, lens, scorer)
        # attn = [(w, s) for w, s in zip(self.vocab.index2word(seq[0][0].tolist()), scores[0].tolist()) if w != 'pad']
        # print(attn)
        return context

    @classmethod
    def get_mask(cls, lens, max_len=None):
        m = max_len if max_len is not None else lens.max().item()
        mask = torch.tensor([[1]*l + [0]*(m-l) for l in lens.tolist()], device=lens.device, dtype=torch.float)
        return mask

    @classmethod
    def run_attn(cls, seq, lens, cond):
        raw_scores = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
        mask = cls.get_mask(lens, max_len=seq.size(1))
        raw_scores -= (1-mask) * 1e20
        scores = F.softmax(raw_scores, dim=1)

        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context, raw_scores
