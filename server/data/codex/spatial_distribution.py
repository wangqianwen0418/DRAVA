#%%
from cProfile import label
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, colors
#%%
dims = {
    'size': 10,
    'orientation': 13
}
foldername = 'HBM622.JXWQ.554'

cells = pd.read_csv(
    f'{foldername}/reg1_stitched_expressions.ome.tiff-cell_centers.csv')

cell_dims = pd.read_csv(
    f'{foldername}/results.csv')


new_cells = pd.concat( [cells['x'], cells['y'], cell_dims['z'].str.split(',', expand=True)] , axis=1)
# %%

def is_small(z):
    return int(z.str.split(',')[dims['size']]) < -1
def is_large(z):
    return int(z.str.split(',')[dims['size']]) > 1
# %%
small_cells = new_cells[new_cells[dims['size']].astype(float) < -2]
large_cells = new_cells[new_cells[dims['size']].astype(float) > 1.2]

fig, ax = plt.subplots()
x = [small_cells['y'].to_list(), large_cells['y'].to_list()]
y = [small_cells['x'].to_list(), large_cells['x'].to_list()]
color = ['pink', 'steelblue']
labels = ['S', 'L']
for i, c in enumerate(color):
    ax.scatter(x[i], [-1*y for y in y[i]], c=color[i], s=10, label=labels[i],
               alpha=0.7, edgecolors='none')
ax.legend()
plt.show()
plt.close()
# %%
h_cells = new_cells[new_cells[dims['orientation']].astype(float) < -1.5]
v_cells = new_cells[new_cells[dims['orientation']].astype(float) > 1.3]


fig, ax = plt.subplots()
x = [h_cells['y'].to_list(), v_cells['y'].to_list()]
y = [h_cells['x'].to_list(), v_cells['x'].to_list()]
color = ['pink', 'steelblue']
labels = ['h', 'v']
for i, c in enumerate(color):
    ax.scatter(x[i], [-1*y for y in y[i]], c=color[i], s=10, label=labels[i],
               alpha=0.7, edgecolors='none')
ax.legend()
plt.show()
plt.close()

# %%
