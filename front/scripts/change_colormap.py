#%%
from PIL import Image, ImageEnhance
from matplotlib import cm as colormap
import numpy as np

import os

#%%
def color_convert(filename, savename=None, mapname='viridis', factor=1):
    if savename == None:
        savename = filename
    cm = colormap.get_cmap(mapname)
    img_src = Image.open(filename).convert('L')

    # increase contrast
    enhancer = ImageEnhance.Contrast(img_src)
    im = enhancer.enhance(factor)


    im = np.array(img_src)
    # im = im + 30 * (im<50) *(im>0)
    cm_im = cm(im)
    cm_im = np.uint8(cm_im * 255)
    cm_im[im==0, :3]=0
    cm_im = Image.fromarray(cm_im)

    cm_im.save(savename)

# apply color map single cell spatial omics
def color_convert_sc(filename, savename=None, mapname='viridis', factor=1):
    if savename == None:
        savename = filename
    cm = colormap.get_cmap(mapname)
    img_src = Image.open(filename).convert('L')

    # increase contrast
    enhancer = ImageEnhance.Contrast(img_src)
    im = enhancer.enhance(factor)


    im = np.array(img_src)
    # im = im + 30 * (im<50) *(im>0)
    cm_im = cm(im)
    cm_im = np.uint8(cm_im * 255)
    cm_im[im==0, :3]=0 # set background to be dark
    # cm_im[im==0, :]=0 # set background to be transparent
    cm_im = Image.fromarray(cm_im)

    cm_im.save(savename)
#%%
file_folder = '../../server/logs/tad/'
for f in os.listdir(file_folder):
    filepath = os.path.join(file_folder, f)
    savepath = filepath.replace('.png', '_color2.png')
    color_convert(filepath, savepath)

# %%
color_convert('../../server/logs/tad/recons_BetaVAE_TAD_1879.png', '../../server/logs/tad/recons_BetaVAE_TAD_1879_color.png',factor=2) 
# %%
