# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from model.emb import Model as Base
from rtfm import featurizer as X


class FILM(nn.Module):
    # from https://arxiv.org/pdf/1806.01946.pdf

    def __init__(self, drnn, dchannel, conv):
        super().__init__()
        self.drnn = drnn
        self.conv = conv
        self.gamma_beta_trans = nn.Linear(3*drnn, 2*dchannel)

    def forward(self, prev, pos):
        TB, dchannel, W, H = prev.size()
        prev = torch.cat([prev, pos], dim=1)
        conv = self.conv(prev).relu()
        return conv


class Model(Base):

    @classmethod
    def create_env(cls, flags, featurizer=None):
        return super().create_env(flags, featurizer=featurizer or X.Concat([X.Text(), X.ValidMoves(), X.RelativePosition()]))

    def __init__(self, observation_shape, num_actions, room_height, room_width, vocab, demb, drnn, drep, pretrained_emb=False, disable_wiki=False):
        super().__init__(observation_shape, num_actions, room_height, room_width, vocab, demb, drnn, drep, pretrained_emb, disable_wiki=disable_wiki)
        self.wiki_rnn = nn.GRU(self.demb, drnn, bidirectional=True, batch_first=True)
        self.task_rnn = nn.GRU(self.demb, drnn, bidirectional=True, batch_first=True)
        self.inv_rnn = nn.GRU(self.demb, drnn, bidirectional=True, batch_first=True)
        self.wiki_scorer = nn.Linear(drnn*2, 1)
        self.task_scorer = nn.Linear(drnn*2, 1)
        self.inv_scorer = nn.Linear(drnn*2, 1)

        self.film1 = FILM(2*drnn, 16, nn.Conv2d(demb+2+3*2*drnn, 16, kernel_size=(3, 3), padding=1))
        self.film2 = FILM(2*drnn, 32, nn.Conv2d(16+2, 32, kernel_size=(3, 3), padding=1))
        self.film3 = FILM(2*drnn, 64, nn.Conv2d(32+2, 64, kernel_size=(3, 3), padding=1))
        self.film4 = FILM(2*drnn, 64, nn.Conv2d(64+2, 64, kernel_size=(3, 3), padding=1))
        self.film5 = FILM(2*drnn, 64, nn.Conv2d(64+2, 64, kernel_size=(3, 3), padding=1))

        self.fc = nn.Sequential(
            nn.Linear(64, self.drep),
            nn.Tanh(),
        )

    def encode_wiki(self, inputs):
        T, B, wiki_len = inputs['wiki'].size()
        if self.disable_wiki:
            return torch.Tensor(T, B, 2*self.drnn).zero_().to(inputs['wiki'].device)
        else:
            x = inputs['wiki'].view(-1, wiki_len).long()
            xlens = inputs['wiki_len'].view(-1)
            return self.run_rnn_selfattn(self.wiki_rnn, x, xlens, self.wiki_scorer)

    def encode_inv(self, inputs):
        T, B, max_len = inputs['inv'].size()
        x = inputs['inv'].view(-1, max_len).long()
        xlens = inputs['inv_len'].view(-1)
        return self.run_rnn_selfattn(self.inv_rnn, x, xlens, self.inv_scorer)

    def encode_task(self, inputs):
        T, B, max_len = inputs['task'].size()
        x = inputs['task'].view(-1, max_len).long()
        xlens = inputs['task_len'].view(-1)
        return self.run_rnn_selfattn(self.task_rnn, x, xlens, self.task_scorer)

    def fuse(self, inputs, cell, inv, wiki, task):
        T, B, H, W, demb = cell.size()
        text = torch.cat([wiki, inv, task], dim=1)
        dtext = text.size(1)
        text = text.unsqueeze(2).unsqueeze(2).expand(T*B, dtext, W, H)

        tb = torch.flatten(cell, 0, 1)  # (T*B, H, W, 3*demb)

        conv_in = tb.transpose(1, 3)  # (T*B, 3*demb, W, H)
        conv_in = torch.cat([conv_in, text], dim=1)
        pos = inputs['rel_pos'].float().view(T*B, H, W, -1).transpose(1, 3)

        c1 = self.film1(conv_in, pos)
        c2 = self.film2(c1, pos)
        c3 = self.film3(c2, pos)
        c4 = self.film4(c3, pos)
        c5 = self.film5(c4+c3, pos)
        conv_out = c5.max(3)[0].max(2)[0]  # pool over spatial dimensions
        flat = conv_out.view(T * B, -1)  # (T*B, -1)
        return self.fc(flat)  # (T*B, drep)
