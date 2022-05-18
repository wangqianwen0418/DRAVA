#%%
import zarr
from tifffile import TiffFile, imread, imwrite
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
import math
import numpy as np
from tqdm import tqdm
# %%
foldername = 'HBM622.JXWQ.554'
tif = imread(f'{foldername}/reg1_stitched_expressions.ome.tif', level=0)
# numpy array, shape (29, 7491, 12664)

tif_mask = imread(f'{foldername}/reg1_stitched_mask.ome.tif')
# 4 indicates cell mask, nuclei masks, cell boundaries, and nucleus boundaries
cell_mask = tif_mask[0]
# numpy array, shape (7491, 12664), number indicates cell id

#%%
# extract pixels fromt tif that has cells
pixels = tif[:, cell_mask!=0]
#%%
# clustering
pixel_cluster_ids = MiniBatchKMeans(n_clusters=7,
                          random_state=0,
                          batch_size=256,
                          max_iter=100).fit_predict(pixels)

cluster_mask = cell_mask.copy()
cluster_mask[cell_mask!=0] = pixel_cluster_ids + 1
# numpy array, shape (7491, 12664), 
# number indicates pixel cluster index (starts from 1)

#%%
# generate cell patches based on pixel cluster
window_size = 128 # 100 -> 128

DOWNSAMPLE_RATIO = 2

cell_centers = pd.read_csv(
    f'{foldername}/reg1_stitched_expressions.ome.tiff-cell_centers.csv')

patch_size = (len(cell_centers), tif.shape[0], math.ceil(
    window_size/DOWNSAMPLE_RATIO), math.ceil(window_size/DOWNSAMPLE_RATIO))

print(f'size of the cell image patches is {patch_size[1:]}')
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

    cluster_patch = np.array(cluster_mask[ y1:y2, x1:x2])
    cell_patch = np.zeros(cluster_patch.shape, dtype=np.float32)
    mask_patch = cell_mask[y1:y2, x1:x2]

    cell_patch[mask_patch == cell_id] = cluster_patch[mask_patch == cell_id]

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

zarr.save(f'{foldername}/cell_patches_{norm_method}.zarr', z)