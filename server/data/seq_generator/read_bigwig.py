#%%
import pyBigWig
import numpy as np



#%%


def getImages(filename, IMG_SIZE, CHR_NUM ):

    CHR = f'chr{CHR_NUM}'
    bw = pyBigWig.open(filename)
    # length of a chromosome
    chr_len = bw.chroms(CHR)

    window_size = 6000
    step_ratio = 0.5
    step = int(window_size * step_ratio)
    # force none type to nan
    records = np.array(bw.stats(CHR, 0, chr_len, nBins = int( chr_len / window_size * IMG_SIZE))).astype('float')
    # convert nan to zero
    records = np.nan_to_num(records)

    min_v = np.nanmin(records)
    max_v = np.nanmax(records)

    #normalize records
    records = (records - min_v)/max_v
    imgs = []
    labels = []


    sample_num = int((chr_len-window_size)/step + 1)
    for i in range(sample_num):
        img = np.ones((IMG_SIZE, IMG_SIZE))
        # stats of a range
        start = i * int( IMG_SIZE * step_ratio ) 
        value_arr = records[start: start + IMG_SIZE]
        for j in range(IMG_SIZE):
            h = int(value_arr[j] * IMG_SIZE)
            img[IMG_SIZE - h:, j] = 0
        imgs.append(img)
        labels.append(np.array([CHR_NUM, int(i * window_size * step_ratio), int(i * window_size * step_ratio + window_size)]))

    imgs = np.stack(imgs)
    labels = np.stack(labels)
    return imgs, labels

#%%
name = "HFFc6_ATAC"

filename = f"./seq_data/{name}.bigWig"
IMG_SIZE = 64

chr_start = 1
chr_end = 8

for CHR_NUM in range(chr_start, chr_end+1):
    imgs, labels = getImages(filename, IMG_SIZE, CHR_NUM )
    if CHR_NUM == chr_start:
        all_labels = labels
        all_imgs = imgs
    else: 
        all_labels = np.concatenate([all_labels, labels])
        all_imgs = np.concatenate([all_imgs, imgs])
    

np.savez(f'../{name}_chr{chr_start}-{chr_end}.npz', imgs=all_imgs, labels=all_labels)