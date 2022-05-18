#%%
import zarr
from tifffile import TiffFile, imread, imwrite
from sklearn.cluster import KMeans
import faiss
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
# numpy shape (num_pixels, num_plex)
pixels = np.transpose(tif[:, cell_mask!=0])
#%%
# clustering
k = 20
n_init = 10
max_iter = 300
kmeans = faiss.Kmeans(d=pixels.shape[1], k=k, niter=max_iter, nredo=n_init)
kmeans.train(pixels.astype(np.float32))
D, I = kmeans.index.search(pixels.astype(np.float32), 1)
pixel_cluster_ids = I.flatten()
#%%
# pixel_cluster_ids = KMeans(n_clusters=7,
#                           random_state=0).fit_predict(pixels)

cluster_mask = cell_mask.copy()
cluster_mask[cell_mask!=0] = pixel_cluster_ids + 1
cluster_mask = cluster_mask.astype('int')
# numpy array, shape (7491, 12664), 
# number indicates pixel cluster index (starts from 1)

#%%
# generate cell patches based on pixel cluster
window_size = 128 # 100 -> 128

DOWNSAMPLE_RATIO = 2

cell_centers = pd.read_csv(
    f'{foldername}/reg1_stitched_expressions.ome.tiff-cell_centers.csv')

patch_size = (len(cell_centers), math.ceil(
    window_size/DOWNSAMPLE_RATIO), math.ceil(window_size/DOWNSAMPLE_RATIO))

print(f'size of the cell image patches is {patch_size[1:]}')
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

    cluster_patch = np.array(cluster_mask[ y1:y2, x1:x2])
    cell_patch = np.zeros(cluster_patch.shape, dtype='int')
    mask_patch = cell_mask[y1:y2, x1:x2]

    cell_patch[mask_patch == cell_id] = cluster_patch[mask_patch == cell_id]

    # padding 0 if smaller than the window size
    ( h, w) = cell_patch.shape
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
    cell_patch = cell_patch[ ::DOWNSAMPLE_RATIO, ::DOWNSAMPLE_RATIO]
    z[idx] = cell_patch

zarr.save(f'{foldername}/cell_patches_{k}cluster.zarr', z)
# %%
from matplotlib import pyplot as plt
import matplotlib

def heatmap(data, row_labels=[], col_labels=[], ax=None, show_colorbar = False,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if show_colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    else:
        cbar = None

    # Show all ticks and label them with the respective list entries.
    if len(col_labels)>0:
        ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    if len(row_labels)>0:
        ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def visualize(cell_id: int):
    z = zarr.open(f'{foldername}/cell_patches_cluster.zarr', mode='r')
    
    cluster_index = [str(i) for i in range(k+1)]
    norm = matplotlib.colors.BoundaryNorm(np.linspace(0, k, k+1), k+1)
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: cluster_index[::-1][norm(x)] if x!=0 else None)

    plt.figure(figsize=(20, 20))
    w = 5
    h = 6
    for i in range(w*h):
        data = z[cell_id + i] # shape of a: 64 x 64
        ax = plt.subplot(5, 6, i + 1)

        im, _ = heatmap(data, cmap=plt.get_cmap('tab20'), norm=norm, ax = ax,
                        # cbar_kw=dict(ticks=np.arange(-3, 4), format=fmt),
                        cbarlabel="pixel cluster index")

    
    
    plt.tight_layout()
    plt.show()


# %%
