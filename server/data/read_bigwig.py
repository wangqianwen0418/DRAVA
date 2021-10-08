#%%
import pyBigWig
import numpy as np

name = "ENCFF158GBQ"

filename = f"./geno_data/{name}.bigWig"
N_BINS = 64
CHR = 'chr7'

#%%
bw = pyBigWig.open(filename)
# length of a chromosome
chr_len = bw.chroms(CHR)

window_size = 12000
step = int(window_size/2)
sample_size = int((chr_len-window_size)/step + 1)
# force none type to nan
records = np.array(bw.stats(CHR, 0, chr_len, nBins = sample_size * N_BINS)).astype('float')
# convert nan to zero
records = np.nan_to_num(records)

min_v = np.nanmin(records)
max_v = np.nanmax(records)

#normalize records
records = (records - min_v)/max_v
imgs = []

for i in range(sample_size):
    img = np.ones((N_BINS, N_BINS))
    # stats of a range
    value_arr = records[i * N_BINS: (i+1) * N_BINS]
    for j in range(N_BINS):
        h = int(value_arr[j] * N_BINS)
        img[N_BINS - h:, j] = 0
    imgs.append(img)
    imgs.append(np.flip(img, 1))

imgs = np.stack(imgs)

np.savez(f'./{name}.npz', imgs=imgs)