#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import csv
import argparse
import numpy as np
import pandas as pd
import termplotlib as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots an experiment in terminal')
    parser.add_argument('exp', help='directory of experiment to plot')
    args = parser.parse_args()

    with open(os.path.join(args.exp, 'logs.csv')) as f:
        header = next(f).strip().split(',')
        rows = [dict(zip(header, l.strip().split(','))) for l in f if not l.startswith('#')]
        df = pd.DataFrame.from_dict(rows)
        # apply smoothing
        df['smoothed_win_rate'] = df['mean_win_rate'].rolling(200, min_periods=1).mean()
    fig = plt.figure()
    fig.plot(df['frames'].to_numpy(np.int), df['smoothed_win_rate'].to_numpy(np.float), width=160, height=80)
    fig.show()
