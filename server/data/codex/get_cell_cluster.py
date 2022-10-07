# %% [markdown]
"""
dataset HBM622.JXWQ.554 downloaded from https://portal.hubmapconsortium.org/browse/dataset/13831dc529085f18ba34e7d29bd41db4
"""

# %%
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
from tifffile import TiffFile, imread, imwrite
import numpy as np
import math
import zarr
import xml.etree.ElementTree as ET
# %%
foldername = 'HBM622.JXWQ.554'

# %%

tif = imread(f'{foldername}/reg1_stitched_expressions.ome.tif', level=0)
# numpy array, shape (29, 7491, 12664)
# 29 indicates 29 antigens
# must use level=0 to indicate the finest level

# get channel names
tags = TiffFile(f'{foldername}/reg1_stitched_expressions.ome.tif').pages[0].tags
xml_description = [t.value for t in tags if t.name == 'ImageDescription'][0]
xml_root = ET.fromstring(xml_description)
channels = [child.attrib['Name'] for child in xml_root[0][1] if 'Name' in child.attrib ]

selected_channels = ['CD107a', 'CD11c', 'ECAD']
selected_channel_index = [channels.index(c) for c in selected_channels ]

w= h = 64
grid_h = math.floor(tif.shape[1]/h) 
grid_w = math.floor(tif.shape[2]/w)
num_imgs = grid_h * grid_w

new_tif = tif[selected_channel_index, :grid_h*h, :grid_w*w]

def split_2d(array, splits):
    h, w = splits
    b = np.stack(np.split(array, w, axis=2), axis=0)
    c = np.concatenate(np.split(b, h, axis=2))
    return c

cell_grids = split_2d(new_tif, (grid_h, grid_w))


zarr.save(f'{foldername}/cell_grids.zarr', cell_grids)
# %%
# visualize


def visualize(cell_id: int):
    # z = zarr.open(f'{foldername}/cell_patches.zarr', mode='r')
    # a = z[cell_id]
    # plt.figure(figsize=(10, 10))
    # for i in range(a.shape[0]):
    #     ax = plt.subplot(5, 6, i + 1)
    #     ax.axis("off")
    #     ax.imshow(a[i])
    # plt.tight_layout()

# %%
