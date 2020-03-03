# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from model.paper_film import Model as Base


class DoubleFILM(nn.Module):
    # from https://arxiv.org/pdf/1806.01946.pdf

    def __init__(self, drnn, demb, dchannel, conv):
        super().__init__()
        self.drnn = drnn
        self.conv = conv
        self.gamma_beta_trans = nn.Linear(2*drnn+2*demb, 2*dchannel)

        self.gamma_beta_conv = nn.Conv2d(conv.in_channels, 2*dchannel, kernel_size=(3, 3), padding=1)
        self.trans_text = nn.Linear(2*drnn+2*demb, dchannel)
        self.image_summ = nn.Linear(dchannel, drnn)

    def forward(self, prev, wiki, inv, task, pos, wiki_inv_attn):
        T, B, *_ = task.size()
        prev = torch.cat([prev, pos], dim=1)
        text = torch.cat([wiki, inv, wiki_inv_attn], dim=1)
        gamma_beta_trans = self.gamma_beta_trans(text)
        gamma, beta = torch.chunk(gamma_beta_trans, 2, dim=1)

        conv = self.conv(prev)
        gamma = gamma.unsqueeze(2).unsqueeze(2).expand_as(conv)
        beta = beta.unsqueeze(2).unsqueeze(2).expand_as(conv)
        image_modulated_with_text = ((1+gamma) * conv + beta).relu()

        gamma_conv, beta_conv = torch.chunk(self.gamma_beta_conv(prev), 2, dim=1)
        text_trans = self.trans_text(text).unsqueeze(2).unsqueeze(2).expand_as(gamma_conv)
        text_modulated_with_image = ((1+gamma_conv) * text_trans + beta_conv).relu()

        mix = image_modulated_with_text + text_modulated_with_image
        image_summ = self.image_summ(mix.max(3)[0].max(2)[0])
        return mix, image_summ


class Model(Base):

    def __init__(self, observation_shape, num_actions, room_height, room_width, vocab, demb, drnn, drnn_small, drep, pretrained_emb=False, disable_wiki=False):
        super().__init__(observation_shape, num_actions, room_height, room_width, vocab, demb, drnn, drnn_small, drep, pretrained_emb, disable_wiki=disable_wiki)
        self.film1 = DoubleFILM(2*drnn, drnn_small, 16, nn.Conv2d(demb+2, 16, kernel_size=(3, 3), padding=1))
        self.film2 = DoubleFILM(2*drnn, drnn_small, 32, nn.Conv2d(16+2, 32, kernel_size=(3, 3), padding=1))
        self.film3 = DoubleFILM(2*drnn, drnn_small, 64, nn.Conv2d(32+2, 64, kernel_size=(3, 3), padding=1))
        self.film4 = DoubleFILM(2*drnn, drnn_small, 64, nn.Conv2d(64+2, 64, kernel_size=(3, 3), padding=1))
        self.film5 = DoubleFILM(2*drnn, drnn_small, 64, nn.Conv2d(64+2, 64, kernel_size=(3, 3), padding=1))
        self.c0_trans = nn.Linear(demb+2, 2*drnn)

        self.wiki_rnn = nn.LSTM(self.demb, drnn, bidirectional=True, batch_first=True)
        self.wiki_rnn2 = nn.LSTM(self.demb, drnn, bidirectional=True, batch_first=True)
        self.task_rnn = self.wiki_rnn
        self.task_scorer = nn.Linear(2*drnn, 1)

    def encode_wiki(self, inputs):
        T, B, wiki_len = inputs['wiki'].size()
        if self.disable_wiki:
            zeros = torch.Tensor(T, B, 2*self.drnn).zero_().to(inputs['wiki'].device)
            return zeros, zeros
        else:
            x = inputs['wiki'].view(-1, wiki_len).long()
            xlens = inputs['wiki_len'].view(-1)
            return self.run_rnn(self.wiki_rnn, x, xlens), self.run_rnn(self.wiki_rnn2, x, xlens)

    def fuse(self, inputs, cell, inv, wiki, task):
        T, B, H, W, demb = cell.size()
        tb = torch.flatten(cell, 0, 1)  # (T*B, H, W, 3*demb)
        pos = inputs['rel_pos'].float().view(T*B, H, W, -1).transpose(1, 3)

        wiki1, wiki2 = wiki
        wiki_lens = inputs['wiki_len'].view(-1).long()
        wiki_attn, _ = self.run_attn(wiki1, wiki_lens, cond=task)

        c0 = tb.transpose(1, 3)  # (T*B, demb, W, H)
        s0 = self.c0_trans(torch.cat([c0, pos], dim=1).max(3)[0].max(2)[0])
        a0, _ = self.run_attn(wiki2, wiki_lens, cond=s0)
        c1, s1 = self.film1(c0, a0, inv, task, pos, wiki_attn)
        a1, _ = self.run_attn(wiki2, wiki_lens, cond=s1)
        c2, s2 = self.film2(c1, a1, inv, task, pos, wiki_attn)
        a2, _ = self.run_attn(wiki2, wiki_lens, cond=s2)
        c3, s3 = self.film3(c2, a2, inv, task, pos, wiki_attn)
        a3, _ = self.run_attn(wiki2, wiki_lens, cond=s3)
        c4, s4 = self.film4(c3, a3, inv, task, pos, wiki_attn)
        a4, _ = self.run_attn(wiki2, wiki_lens, cond=s4)
        c5, s5 = self.film5(c4+c3, a4, inv, task, pos, wiki_attn)
        conv_out = c5.max(3)[0].max(2)[0]  # pool over spatial dimensions
        flat = conv_out.view(T * B, -1)  # (T*B, -1)
        return self.fc(flat)  # (T*B, drep)
