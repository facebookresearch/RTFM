# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from model.reader import Model as Base


class Model(Base):

    def __init__(self, observation_shape, num_actions, room_height, room_width, vocab, demb, drnn, drep, pretrained_emb=False, disable_wiki=False):
        super().__init__(observation_shape, num_actions, room_height, room_width, vocab, demb, drnn, drep, pretrained_emb, disable_wiki=disable_wiki)
        self.conv = nn.Sequential(
            nn.Conv2d(3*demb, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, self.dconv_out, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.dconv_out*self.room_height_conv_out*self.room_width_conv_out, self.drep),
            nn.Tanh(),
        )

    def encode_wiki(self, inputs):
        T, B, wiki_len = inputs['wiki'].size()
        if self.disable_wiki:
            return torch.Tensor(T, B, self.demb).zero_().to(inputs['wiki'].device)
        else:
            return self.emb(inputs['wiki'].long()).sum(-2)

    def encode_inv(self, inputs):
        T, B, inv_len = inputs['inv'].size()
        return self.emb(inputs['inv'].long()).sum(-2)

    def encode_cell(self, inputs):
        T, B, H, W, n_placement, name_len = inputs['name'].size()
        placement = self.emb(inputs['name'].long()).sum(-2)
        return placement.sum(4)

    def fuse(self, inputs, cell, inv, wiki, task):
        T, B, H, W, demb = cell.size()
        grid = torch.cat([
            cell,
            inv.unsqueeze(2).unsqueeze(3).expand(T, B, H, W, inv.size(-1)),
            wiki.unsqueeze(2).unsqueeze(3).expand(T, B, H, W, wiki.size(-1)),
        ], dim=4)  # (T, B, H, W, 2*demb)

        tb = torch.flatten(grid, 0, 1)  # (T*B, H, W, 3*demb)

        conv_in = tb.transpose(1, 3)  # (T*B, 3*demb, W, H)
        conv_out = self.conv(conv_in)
        flat = conv_out.view(T * B, -1)  # (T*B, -1)
        return self.fc(flat)  # (T*B, drep)
