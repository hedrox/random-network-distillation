import matplotlib.pyplot as plt
import pickle
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Visualize attention')
parser.add_argument("video_path", type=str, help="Path to the videos_0.pk path in the openai_savings")

args = parser.parse_args()
path = args.video_path
f = open(path, 'rb')

for _ in range(200):
    episode = pickle.load(f)

episode = pickle.load(f)

attention = episode['attention']
obs = episode['obs']

square = 7
ix = 1

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(14, 7)


for i in range(square):
    for j in range(square):
        # specify subplot and turn of axis
        sub = fig.add_subplot(gs[i,j])
        sub.set_xticks([])
        sub.set_yticks([])
        sub.imshow(attention[20, :, :, ix-1], cmap='gray', aspect='auto')
        ix += 1

sub = fig.add_subplot(gs[7:, :])
sub.set_xticks([])
sub.set_yticks([])
sub.imshow(obs[20,...,:3])

plt.show()

# square = 7
# ix = 1
# for _ in range(square):
#     for _ in range(square):
#         # specify subplot and turn of axis
#         ax = plt.subplot(square, square, ix)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         # plot filter channel in grayscale
#         plt.imshow(attention[480, :, :, ix-1], cmap='gray')
#         ix += 1

# plt.show()

# imgplot = plt.imshow(obs[480,...,:3])
# plt.show()
