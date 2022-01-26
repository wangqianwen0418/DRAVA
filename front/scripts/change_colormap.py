#%%
from PIL import Image
from matplotlib import cm as colormap
import numpy as np

import os

#%%
def color_convert(filename, mapname='viridis'):
    cm = colormap.get_cmap(mapname)
    img_src = Image.open(filename).convert('L')
    img_src.thumbnail((512,512))
    im = np.array(img_src)
    im = cm(im)
    im = np.uint8(im * 255)
    im = Image.fromarray(im)
    im.save(filename)

file_folder = '../public/assets/matrix_simu'
for f in os.listdir(file_folder):
    filepath = os.path.join(file_folder, f)
    color_convert(filepath)

# %%
