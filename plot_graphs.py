import numpy as np
import pandas as pd
import os, sys, argparse

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--rnd_path", type=str, help="Path to the progress.csv of the RND run")
parser.add_argument("--aarnd_path", type=str,
                    help="Path to the progress.csv of the AA RND run")
parser.add_argument("--egornd_path", type=str,
                    help="Path to the progress.csv of the Ego RND run")
args = parser.parse_args()

data1 = pd.read_csv(args.rnd_path)
data2 = pd.read_csv(args.aarnd_path)
data3 = pd.read_csv(args.egornd_path)

data = [data1, data2, data3]
for idx in range(len(data)):
    twomillion = 200000000
    while True:
        res = data[idx]['tcount'][data[idx]['tcount'] == twomillion]
        if not res.empty:
            data[idx] = data[idx][:res.index[0]]
            break
        else:
            twomillion += 1

data1, data2, data3 = data

#fig, axes = plt.subplots(figsize=(19.20, 10.80), nrows=2, ncols=2)
fig, axes = plt.subplots(nrows=2, ncols=2)

"""
retextmean, retextstd, retintmean, retintstd, rewintmean_norm, rewintmean_unnorm,
vpredextmean, vpredintmean are interesting metrics
"""

#fig.suptitle("Montezuma's Revenge Ego vs AA-RND", fontsize=10,y=0.9,x=0.51)
data1.plot(x='tcount', y='rewtotal', ax=axes[0,0], color='red', label='RND')
data2.plot(x='tcount', y='rewtotal', ax=axes[0,0], color='green', label='AA RND')
data3.plot(x='tcount', y='rewtotal', ax=axes[0,0], color='blue', label='Ego RND')
axes[0,0].set_xlabel('Frames')
axes[0,0].set_ylabel('Total Rewards')


data1.plot(x='tcount', y='n_rooms', ax=axes[0,1], color='red', label='RND')
data2.plot(x='tcount', y='n_rooms', ax=axes[0,1], color='green', label='AA RND')
data3.plot(x='tcount', y='n_rooms', ax=axes[0,1], color='blue', label='Ego RND')
axes[0,1].set_xlabel('Frames')
axes[0,1].set_ylabel('No Rooms')


data1.plot(x='tcount', y='eprew', ax=axes[1,0], color='red', label='RND')
data2.plot(x='tcount', y='eprew', ax=axes[1,0], color='green', label='AA RND')
data3.plot(x='tcount', y='eprew', ax=axes[1,0], color='blue', label='Ego RND')
axes[1,0].set_xlabel('Frames')
axes[1,0].set_ylabel('Episodic Rewards')


data1.plot(x='tcount', y='best_ret', ax=axes[1,1], color='red', label='RND')
data2.plot(x='tcount', y='best_ret', ax=axes[1,1], color='green', label='AA RND')
data3.plot(x='tcount', y='best_ret', ax=axes[1,1], color='blue', label='Ego RND')
axes[1,1].set_xlabel('Frames')
axes[1,1].set_ylabel('Best Rewards')

#fig.show()
#plt.show()

plot_name = 'montezuma-all-three'
plt.tight_layout()
plt.savefig(f'{plot_name}.eps')

