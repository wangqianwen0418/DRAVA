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
import os
import zarr
# %%
foldername = 'HBM622.JXWQ.554'
# %%

tif = imread(f'{foldername}/reg1_stitched_expressions.ome.tif', level=0)
# numpy array, shape (29, 7491, 12664)
# 29 indicates 29 antigens
# must use level=0 to indicate the finest level
channel_max = [tif[i].max() for i in range(tif.shape[0])]

tif_mask = imread(f'{foldername}/reg1_stitched_mask.ome.tif')
# numpy array, shape (4, 7491, 12664)
# 4 indicates cell mask, nuclei masks, cell boundaries, and nucleus boundaries
cell_mask = tif_mask[0]
# %%
##############################
# determine window size
##############################


def getLargeSlice(array):
    array = array[array != 0]
    if len(array) == 0:
        return 0
    (cell_ids, counts) = np.unique(array, return_counts=True)
    return max(counts)


widths = [getLargeSlice(cell_mask[i, :]) for i in range(cell_mask.shape[0])]
width = max(widths)

heights = [getLargeSlice(cell_mask[:, i]) for i in range(cell_mask.shape[1])]
height = max(heights)

window_size = max(width, height)
window_size = 128 # 100 -> 128

# %%
##############################
# cut cells
##############################
DOWNSAMPLE_RATIO = 2

cell_centers = pd.read_csv(
    f'{foldername}/reg1_stitched_expressions.ome.tiff-cell_centers.csv')

patch_size = (len(cell_centers), tif.shape[0], math.ceil(
    window_size/DOWNSAMPLE_RATIO), math.ceil(window_size/DOWNSAMPLE_RATIO))

print(
    f'size of the cell image patches is {patch_size[1:]}')

# patches = np.zeros((len(cell_centers), tif.shape[0], window_size, window_size))

# %%

z = zarr.zeros(patch_size, chunks=(
    100, None, None, None), dtype='float32')

for idx, row in tqdm(cell_centers.iterrows()):
    if row['ID'] == 0:
        continue

    x = int(row['y'])  # x y are switched in the csv file
    x1 = max(0, int(x-window_size/2))
    x2 = min(cell_mask.shape[1]-1, int(x + window_size/2))

    y = int(row['x'])
    y1 = max(0, int(y - window_size/2))
    y2 = min(cell_mask.shape[0]-1, int(y + window_size/2))

    cell_id = row['ID']

    tif_patch = np.array(tif[:, y1:y2, x1:x2])
    cell_patch = np.zeros(tif_patch.shape, dtype=np.float32)
    mask_patch = cell_mask[y1:y2, x1:x2]

    # normalize values
    for i, v in enumerate(channel_max):
        cell_patch[i, mask_patch == cell_id] = tif_patch[i,
                                                         mask_patch == cell_id]/v

    # padding 0 if smaller than the window size
    (c, h, w) = cell_patch.shape
    if w < window_size or h < window_size:
        cell_patch = np.pad(
            cell_patch, (
                (0, 0),
                (math.floor((window_size - h)/2), math.ceil((window_size - h)/2)),
                (math.floor((window_size - w)/2), math.ceil((window_size - w)/2))
            )
        )

    if cell_patch.max() == 0:
        print(cell_id, 'empty')
        break

    # downsampling
    cell_patch = cell_patch[:, ::DOWNSAMPLE_RATIO, ::DOWNSAMPLE_RATIO]
    z[idx] = cell_patch

zarr.save(f'{foldername}/cell_patches.zarr', z)
# %%
# visualize


def visualize(cell_id: int):
    z = zarr.open(f'{foldername}/cell_patches.zarr', mode='r')
    a = z[cell_id]
    plt.figure(figsize=(10, 10))
    for i in range(a.shape[0]):
        ax = plt.subplot(5, 6, i + 1)
        ax.axis("off")
        ax.imshow(a[i])
    plt.tight_layout()

# %%
