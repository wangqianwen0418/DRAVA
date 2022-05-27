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
# %%
foldername = 'HBM622.JXWQ.554'

tif_mask = imread(f'{foldername}/reg1_stitched_mask.ome.tif')
# numpy array, shape (4, 7491, 12664)
# 4 indicates cell mask, nuclei masks, cell boundaries, and nucleus boundaries
cell_mask = tif_mask[0]
nucleus_mask = tif_mask[1]
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

# each patch has three channels, 0: background, 1: cell but not nuclei, 2: nuclei
patch_size = (len(cell_centers), math.ceil(
    window_size/DOWNSAMPLE_RATIO), math.ceil(window_size/DOWNSAMPLE_RATIO))

print(
    f'size of the cell image patches is {patch_size[1:]}')

# patches = np.zeros((len(cell_centers), tif.shape[0], window_size, window_size))

# %%

z = zarr.zeros(patch_size, chunks=(
    100, None, None), dtype='int')

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

    
    cell_mask_patch = cell_mask[y1:y2, x1:x2]
    nucleus_mask_patch = nucleus_mask[y1:y2, x1:x2]

    cell_patch = np.zeros(cell_mask_patch.shape, dtype='int')

    cell_patch[cell_mask_patch==cell_id] = 1
    cell_patch[nucleus_mask_patch==cell_id] = 2

    # padding 0 if smaller than the window size
    (h, w) = cell_patch.shape
    if w < window_size or h < window_size:
        cell_patch = np.pad(
            cell_patch, (
                (math.floor((window_size - h)/2), math.ceil((window_size - h)/2)),
                (math.floor((window_size - w)/2), math.ceil((window_size - w)/2))
            )
        )

    if cell_patch.max() == 0:
        print(cell_id, 'empty')
        break

    # downsampling
    cell_patch = cell_patch[::DOWNSAMPLE_RATIO, ::DOWNSAMPLE_RATIO]
    z[idx] = cell_patch

zarr.save(f'{foldername}/cell_patches_cell+nucleus.zarr', z)
# %%
# visualize
from matplotlib import pyplot, colors

def visualize(cell_id: int, figsize=20):
    z = zarr.open(f'{foldername}/cell_patches_cell+nucleus.zarr', mode='r')

    k = z.shape[1]
    cmap=pyplot.get_cmap('tab10')
    norm = colors.BoundaryNorm(boundaries = [ i-0.1 for i in range(0, k+2)], ncolors=k+1)
    pyplot.figure(figsize=(figsize, figsize))
    w = 5
    h = 6
    for i in range(w*h):
        data = z[cell_id + i]  # shape of a: 64 x 64

        alpha = np.ones(data.shape)
        alpha[data == 0] = 0  # make bacground trasparent

        ax = pyplot.subplot(5, 6, i + 1)

        ax.imshow(data, alpha=alpha, cmap = cmap, norm = norm)
        ax.axis("off")

    pyplot.tight_layout()
    pyplot.show()

# %%
