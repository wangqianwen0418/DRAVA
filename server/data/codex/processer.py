# %% [markdown]
"""
dataset HBM622.JXWQ.554 downloaded from https://portal.hubmapconsortium.org/browse/dataset/13831dc529085f18ba34e7d29bd41db4
"""

# %%
import pandas as pd
from tifffile import TiffFile, imread, imwrite
import numpy as np
import math
# %%
foldername = 'HBM622.JXWQ.554'
# %%

tif = imread(f'{foldername}/reg1_stitched_expressions.ome.tif')
# numpy array, shape (29, 7491, 12664)
# 29 indicates 29 antigens

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
print(f'size of the cell image patches is {window_size}')

# %%
##############################
# cut cells
##############################
cell_centers = pd.read_csv(
    f'{foldername}/reg1_stitched_expressions.ome.tiff-cell_centers.csv')

patches = []

for idx, row in cell_centers.iterrows():
    if row['ID'] == 0:
        continue

    x = row['x']
    x1 = max(0, int(x-window_size/2))
    x2 = min(cell_mask.shape[1]-1, int(x + window_size/2))

    y = row['y']
    y1 = max(0, int(y - window_size/2))
    y2 = min(cell_mask.shape[0]-1, int(y + window_size/2))

    cell_id = row['ID']

    tif_patch = tif[:, y1:y2, x1:x2]
    mask_patch = cell_mask[y1:y2, x1:x2]
    maskout = np.array(
        [mask_patch != cell_id for i in range(tif_patch.shape[0])])
    tif_patch[maskout] = 0
    # padding 0 if smaller than the window size
    (c, h, w) = tif_patch.shape
    if h == 0:
        print(x1, x2)
        break
    if w < window_size or h < window_size:

        tif_patch = np.pad(
            tif_patch, (
                (0, 0),
                (math.floor((window_size - h)/2), math.ceil((window_size - h)/2)),
                (math.floor((window_size - w)/2), math.ceil((window_size - w)/2))
            )
        )
    patches.append(tif_patch)

patches = np.stack(patches, axis=0)

# %%
np.save(f'{foldername}/cells.npy', patches)

# %%
