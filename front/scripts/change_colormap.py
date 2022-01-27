#%%
from PIL import Image, ImageEnhance
from matplotlib import cm as colormap
import numpy as np

import os

#%%
def color_convert(filename, savename=None, mapname='viridis', factor=1.5):
    if savename == None:
        savename = filename
    cm = colormap.get_cmap(mapname)
    img_src = Image.open(filename).convert('L')

    # increase contrast
    enhancer = ImageEnhance.Contrast(img_src)
    im = enhancer.enhance(factor)


    im = np.array(im)
    im = cm(im)
    im = np.uint8(im * 255)
    im = Image.fromarray(im)

    im.save(savename)

#%%
file_folder = '../public/assets/matrix_simu_copy'
for f in os.listdir(file_folder):
    filepath = os.path.join(file_folder, f)
    savepath = filepath.replace('/matrix_simu_copy/', '/matrix_simu/')
    color_convert(filepath, savepath)

# %%
