# %% [markdown]
"""
dataset HBM622.JXWQ.554 downloaded from https://portal.hubmapconsortium.org/browse/dataset/13831dc529085f18ba34e7d29bd41db4
"""

# %%
from unittest.mock import patch
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
from tifffile import TiffFile, imread, imwrite
import numpy as np
import math
import zarr
import xml.etree.ElementTree as ET
from PIL import Image

# %%
foldername = 'HBM622.JXWQ.554'
selected_channels = {'CD15': [55,27851], 'CD31':[45,2186], 'ECAD': [59,12306]} # channel names and the data domains
zoom_level = 1 # 0 indicates the most detailed view
w = h = 64
shift_step = 20

#%%
def split_2d(array, splits):
    h, w = splits
    b = np.stack(np.split(array, w, axis=2), axis=0)
    c = np.concatenate(np.split(b, h, axis=2))
    return c

# %%

# numpy array, shape (29, 7491, 12664)
# 29 indicates 29 antigens
# must use level=0 to indicate the finest level
tif = imread(f'{foldername}/reg1_stitched_expressions.ome.tif', level=zoom_level)

# get channel names
tags = TiffFile(f'{foldername}/reg1_stitched_expressions.ome.tif').pages[0].tags
xml_description = [t.value for t in tags if t.name == 'ImageDescription'][0]
xml_root = ET.fromstring(xml_description)
channels = [child.attrib['Name'] for child in xml_root[0][1] if 'Name' in child.attrib ]
selected_channel_index = [channels.index(c) for c in selected_channels ]


grid_h = math.floor(tif.shape[1]/h) 
grid_w = math.floor(tif.shape[2]/w)

new_tif = tif[selected_channel_index, :grid_h*h, :grid_w*w]
cell_grids = split_2d(new_tif, (grid_h, grid_w))


for shift in range(0, min(w, h), shift_step):

    new_tif_shift = tif[selected_channel_index, shift:(grid_h-1)*h + shift, shift:(grid_w-1)*w+shift]
    grids_shift = split_2d(new_tif_shift, (grid_h-1, grid_w-1))
    cell_grids = np.concatenate( (cell_grids, grids_shift), axis=0)

# normalize
patch_size = (cell_grids.shape[0], len(selected_channel_index), h, w)
norm_cell_grids = zarr.zeros(patch_size, chunks=(
    1000, None, None, None), dtype='float32')
for i,c in enumerate(selected_channels):
    v_min, v_max = selected_channels[c]
    clip_grids = np.clip(cell_grids[:, i, :, :], v_min, v_max)
    norm_cell_grids[:, i, :, :] = (clip_grids- v_min) / (v_max - v_min)

zarr.save(f'{foldername}/cell_grids_level{zoom_level}_step{shift_step}.zarr', norm_cell_grids)

#%% for each item, save them cell boundaries
tif_mask = imread(f'{foldername}/reg1_stitched_mask.ome.tif', level=zoom_level)
# 4 indicates cell mask, nuclei masks, cell boundaries, and nucleus boundaries
cell_boundries = tif_mask[2:3]

cell_boundries = cell_boundries[:, :grid_h*h, :grid_w*w]

cell_boundries_grids = split_2d(cell_boundries, (grid_h, grid_w))
for shift in range(0, min(w, h), shift_step):
    cell_boundries_shift = cell_boundries[:, shift:(grid_h-1)*h + shift, shift:(grid_w-1)*w+shift]
    cell_boundries_grids_shift = split_2d(cell_boundries_shift, (grid_h-1, grid_w-1))
    cell_boundries_grids = np.concatenate( (cell_boundries_grids, cell_boundries_grids_shift), axis=0)

zarr.save(f'{foldername}/cell_masks_level{zoom_level}_step{shift_step}.zarr', cell_boundries_grids)
# %%
# visualize


def visualize(item_id: int):
    z = zarr.open(f'{foldername}/cell_grids_level{zoom_level}_step{shift_step}.zarr', mode='r')
    a = z[item_id]

    masks = zarr.open(f'{foldername}/cell_masks_level{zoom_level}_step{shift_step}.zarr', mode='r')
    mask = masks[item_id]

    plt.figure(figsize=(10, 10))
    
    # a subfigure for each antigen chanel
    for i in range(a.shape[0]):
        ax = plt.subplot(5, 6, i + 1)
        ax.axis("off")
        rgb_img = np.zeros(a.transpose(1, 2, 0).shape)
        rgb_img[:,:, i] = a[i,:,:]

        ax.imshow(rgb_img)
    
    # a subfigure for three channels
    ax = plt.subplot(5, 6, a.shape[0]+1)
    ax.axis("off")
    ax.imshow(a.transpose(1, 2, 0))

    # a subfigure for cell bounders
    ax = plt.subplot(5, 6, a.shape[0]+2)
    ax.axis("off")
    a[:, mask[0]>0] = 1
    ax.imshow(a.transpose(1, 2, 0))
    plt.tight_layout()

# %%
visualize(4000)
visualize(5000)

# %%
