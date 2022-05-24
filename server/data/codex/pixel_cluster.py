# %%
import umap.umap_ as umap
import zarr
from tifffile import TiffFile, imread, imwrite
from sklearn.cluster import KMeans
import faiss
import pandas as pd
import math
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot, colors

# %%
foldername = 'HBM622.JXWQ.554'
tif = imread(f'{foldername}/reg1_stitched_expressions.ome.tif', level=0)
# numpy array, shape (29, 7491, 12664)

tif_mask = imread(f'{foldername}/reg1_stitched_mask.ome.tif')
# 4 indicates cell mask, nuclei masks, cell boundaries, and nucleus boundaries
cell_mask = tif_mask[0]
# numpy array, shape (7491, 12664), number indicates cell id

# %%
# extract pixels fromt tif that has cells
# numpy shape (num_pixels, num_plex)
pixels = np.transpose(tif[:, cell_mask != 0])
##################


# %%
# clustering
k = 9
n_init = 10
max_iter = 300
kmeans = faiss.Kmeans(d=pixels.shape[1], k=k, niter=max_iter, nredo=n_init)
kmeans.train(pixels.astype(np.float32))
D, I = kmeans.index.search(pixels.astype(np.float32), 1)
I = I + 1  # starts fron index 1, 0 is used to indicate background
pixel_cluster_ids = I.flatten()

cluster_idxs, cluster_counts = np.unique(pixel_cluster_ids, return_counts=True)
##################

# %%
# random sampling from all cells
sample_size = 1000
idx = np.random.randint(0, pixel_cluster_ids.shape[0], size=sample_size)

#sample & UMAP

reducer = umap.UMAP()
embedding = reducer.fit_transform(pixels[idx])
umap_points = np.concatenate((embedding, I[idx]), axis=1)

# %%
# pca

mt = pixels.astype('float32')
mat = faiss.PCAMatrix(29, 2)
mat.train(mt)
assert mat.is_trained
tr = mat.apply_py(mt)
points = np.concatenate((tr, I), axis=1)


pca_points = points[idx, :]

# %% draw chart
cmap = pyplot.get_cmap('tab10')
norm = colors.BoundaryNorm(
    boundaries=[i-0.1 for i in range(0, k+2)], ncolors=k+1)

fig, (ax1, ax2, ax3) = pyplot.subplots(3, 1, figsize=(4, 12))
# draw scatter
scatter = ax1.scatter(x=pca_points[:, 0], y=pca_points[:, 1],
                      s=2.1, c=pca_points[:, 2], cmap=cmap, norm=norm)
legend = ax1.legend(*scatter.legend_elements(),
                    loc="lower right", title="clusters")
ax1.add_artist(legend)
# ax1.axis("off")
ax1.xaxis.set_ticklabels([])
ax1.yaxis.set_ticklabels([])

scatter2 = ax2.scatter(x=umap_points[:, 0], y=umap_points[:, 1],
                       s=2.1, c=umap_points[:, 2], cmap=cmap, norm=norm)


# draw histogram
bar = ax3.bar(
    x=cluster_idxs,
    height=cluster_counts,
    color=cmap.colors[1:]  # save colors[0] for background
)

pyplot.show()
#############


# %%

cluster_mask = cell_mask.copy()
cluster_mask[cell_mask != 0] = pixel_cluster_ids
cluster_mask = cluster_mask.astype('int')
# numpy array, shape (7491, 12664),
# number indicates pixel cluster index (starts from 1)

# %%
# generate cell patches based on pixel cluster
window_size = 128  # 100 -> 128

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

    cluster_patch = np.array(cluster_mask[y1:y2, x1:x2])
    cell_patch = np.zeros(cluster_patch.shape, dtype='int')
    mask_patch = cell_mask[y1:y2, x1:x2]

    cell_patch[mask_patch == cell_id] = cluster_patch[mask_patch == cell_id]

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

zarr.save(f'{foldername}/cell_patches_{k}cluster.zarr', z)
# %%


def visualize(cell_id: int, figsize=20):
    z = zarr.open(f'{foldername}/cell_patches_{k}cluster.zarr', mode='r')

    # make value discrete and norm into [0, 1]
    norm = colors.BoundaryNorm(boundaries=list(range(k+1)), ncolors=k+1)

    pyplot.figure(figsize=(figsize, figsize))
    w = 5
    h = 6
    for i in range(w*h):
        data = z[cell_id + i]  # shape of a: 64 x 64

        alpha = np.ones(data.shape)
        alpha[data == 0] = 0  # make bacground trasparent

        ax = pyplot.subplot(5, 6, i + 1)

        ax.imshow(data, cmap=cmap, norm=norm, alpha=alpha)
        ax.axis("off")

    pyplot.tight_layout()
    pyplot.show()


# %%
