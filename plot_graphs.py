import numpy as np
import pandas as pd
import sys, argparse

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("model1_path", type=str, help="Path to the progress.csv of the first model")
parser.add_argument("model2_path", type=str,
                    help="Path to the progress.csv of the second model being compared to")
args = parser.parse_args()

data1 = pd.read_csv(args.model1_path)
data2 = pd.read_csv(args.model2_path)

fig, axes = plt.subplots(nrows=2, ncols=2)

"""
retextmean, retextstd, retintmean, retintstd, rewintmean_norm, rewintmean_unnorm,
vpredextmean, vpredintmean are interesting metrics
"""

data1.plot(x='tcount', y='rewtotal', ax=axes[0,0], color='blue')
data2.plot(x='tcount', y='rewtotal', ax=axes[0,0], color='red')

data1.plot(x='tcount', y='n_rooms', ax=axes[0,1], color='blue')
data2.plot(x='tcount', y='n_rooms', ax=axes[0,1], color='red')

data1.plot(x='tcount', y='eprew', ax=axes[1,0], color='blue')
data2.plot(x='tcount', y='eprew', ax=axes[1,0], color='red')

data1.plot(x='tcount', y='best_ret', ax=axes[1,1], color='blue')
data2.plot(x='tcount', y='best_ret', ax=axes[1,1], color='red')

fig.show()
plt.show()

