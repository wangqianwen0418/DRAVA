# %% [markdown]
"""
dataset HBM622.JXWQ.554 downloaded from https://portal.hubmapconsortium.org/browse/dataset/13831dc529085f18ba34e7d29bd41db4
"""

# %%
import pandas as pd
from tqdm import tqdm
from tifffile import TiffFile, imread, imwrite
import numpy as np
import math
import os
# %%
foldername = 'HBM622.JXWQ.554'
# %%

tif = imread(f'{foldername}/reg1_stitched_expressions.ome.tif', level=0)
# numpy array, shape (29, 7491, 12664)
# 29 indicates 29 antigens
# must use level=0 to indicate the finest level

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
print(f'size of the cell image patches is ({tif.shape[0]}, {window_size}, {window_size})')

# %%
##############################
# cut cells
##############################
cell_centers = pd.read_csv(
    f'{foldername}/reg1_stitched_expressions.ome.tiff-cell_centers.csv')

# patches = np.zeros((len(cell_centers), tif.shape[0], window_size, window_size))

#%%

patches_folder = f'{foldername}/cells/'
if not os.path.exists(patches_folder):
    os.mkdir(patches_folder)

for idx, row in tqdm(cell_centers.iterrows()):
    if row['ID'] == 0:
        continue

    x = int(row['y']) # x y are switched in the csv file
    x1 = max(0, int(x-window_size/2))
    x2 = min(cell_mask.shape[1]-1, int(x + window_size/2))

    y = int(row['x'])
    y1 = max(0, int(y - window_size/2))
    y2 = min(cell_mask.shape[0]-1, int(y + window_size/2))

    cell_id = row['ID']

    tif_patch = np.array(tif[:, y1:y2, x1:x2]) # deep copy
    mask_patch = cell_mask[y1:y2, x1:x2]

    tif_patch[:, mask_patch != cell_id] = 0

    # padding 0 if smaller than the window size
    (c, h, w) = tif_patch.shape
    if w < window_size or h < window_size:
        tif_patch = np.pad(
            tif_patch, (
                (0, 0),
                (math.floor((window_size - h)/2), math.ceil((window_size - h)/2)),
                (math.floor((window_size - w)/2), math.ceil((window_size - w)/2))
            )
        )

    np.save(f'{patches_folder}/cell_{idx}.npy', tif_patch)

# %%
# visualize
from matplotlib import pyplot as plt
cell_id = 2
a = np.load(f'{patches_folder}/cell_{cell_id}.npy')
plt.figure(figsize=(10, 10))
for i in range(29):
    ax = plt.subplot(5, 6, i + 1)
    ax.axis("off")
    ax.imshow(a[i])
plt.tight_layout()