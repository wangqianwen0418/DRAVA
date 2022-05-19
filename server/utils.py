import pytorch_lightning as pl
from matplotlib import pyplot, colors
import numpy as np

## Utils to handle newer PyTorch Lightning changes from version 0.6
## ==================================================================================================== ##


def data_loader(fn):
    """
    Decorator to handle the deprecation of data_loader from 0.7
    :param fn: User defined data loader function
    :return: A wrapper for the data_loader function
    """

    def func_wrapper(self):
        try: # Works for version 0.6.0
            return pl.data_loader(fn)(self)

        except: # Works for version > 0.6.0
            return fn(self)

    return func_wrapper

def drawMasks(masks, figsize, nrows, save_path):
    '''
    save multiclass masks to images
    :param masks: [batch, num_class, H, W]
    :param figsize: int, size of each subfigure
    :param nrows: number of images at each row
    '''
    k = masks.shape[1] # number of classes
    num_img = masks.shape[0]
    ncols = num_img//nrows + 1 # number of images at each col

    cmap=pyplot.get_cmap('tab10')
    norm = colors.BoundaryNorm(boundaries = list(range(k+1)), ncolors=k+1)
    pyplot.figure(figsize=(figsize, figsize))

    for i in range(num_img):
            data = masks[i] # shape [num_class, h, w]
            data = np.argmax(data, axis=0) # convert one hot into cluster indices

            alpha = np.ones(data.shape)
            alpha[data==0] = 0 # make bacground trasparent

            ax = pyplot.subplot(ncols, nrows, i + 1)

            ax.imshow(data, cmap=cmap, norm=norm, alpha = alpha)
            ax.axis("off")
 
    pyplot.tight_layout()
    pyplot.savefig(save_path)
    pyplot.close('all')