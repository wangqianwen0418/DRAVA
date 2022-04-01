#%%
import pandas as pd
filename = '../public/assets/results_chr1-5_10k_onTad.csv'
df = pd.read_csv(filename.replace('.csv', '_origin.csv'))

#%%
# add ctct values at edge

import pyBigWig
bw = pyBigWig.open('./HFFC6_CTCF.mRp.clN.bigWig')

col_ctcf_left = []
col_ctcf_right = []
col_ctcf_v = []

for index, row in df.iterrows():
    chr, start, end, *_ = row
    # ctcf value at the left edge
    try:
        ctcf_left = bw.stats(f'chr{int(chr)}', int(start)*10000-5000, int(start)*10000+5000)[0]
        ctcf_right = bw.stats(f'chr{int(chr)}', int(end)*10000-5000, int(end)*10000+5000)[0]
        if not ctcf_left:
            ctcf_left = 0
        if not ctcf_right:
            ctcf_right = 0
    except:
        ctcf_left = 0
        ctcf_right = 0
    ctcf_v = (ctcf_left +ctcf_right)/2
    col_ctcf_left.append(ctcf_left)
    col_ctcf_right.append(ctcf_right)
    col_ctcf_v.append(ctcf_v)

df['ctcf_mean'] = col_ctcf_v
df['ctcf_left'] = col_ctcf_left
df['ctcf_right'] = col_ctcf_right
df.to_csv(filename, index=False)
# %%
bw = pyBigWig.open('./HFFc6_ATAC.bigWig')

col_atac_left = []
col_atac_right = []
col_atac_v = []

for index, row in df.iterrows():
    chr, start, end, *_ = row
    # ctcf value at the left edge
    try:
        ctcf_left = bw.stats(f'chr{int(chr)}', int(start)*10000-5000, int(start)*10000+5000)[0]
        ctcf_right = bw.stats(f'chr{int(chr)}', int(end)*10000-5000, int(end)*10000+5000)[0]
        if not ctcf_left:
            ctcf_left = 0
        if not ctcf_right:
            ctcf_right = 0
    except:
        ctcf_left = 0
        ctcf_right = 0
    ctcf_v = (ctcf_left +ctcf_right)/2
    col_atac_left.append(ctcf_left)
    col_atac_right.append(ctcf_right)
    col_atac_v.append(ctcf_v)

df['atac_mean'] = col_atac_v
df['atac_left'] = col_atac_left
df['atac_right'] = col_atac_right
df.to_csv(filename, index=False)
# %%
celeba_df = pd.read_csv('../public/assets/results_celeba.csv')
label_df = pd.read_csv('../../server/data/celeba/list_attr_celeba.txt', sep=r"\s+")
label_df = label_df.loc[:len(celeba_df)]
# add gender
celeba_df['gender'] = label_df['Male'].apply(lambda x: 'M' if x== 1 else 'F')
# hair color
celeba_df['hair'] = 'unknown'
celeba_df.loc[ label_df['Black_Hair'] == 1, 'hair'] = 'black'
celeba_df.loc[ label_df['Brown_Hair'] == 1, 'hair'] = 'brown'
celeba_df.loc[ label_df['Gray_Hair'] == 1, 'hair'] = 'gray'
celeba_df.loc[ label_df['Blond_Hair'] == 1, 'hair'] = 'blond'
# 
celeba_df['smiling'] = label_df['Smiling'].apply(lambda x: 'Y' if x== 1 else 'N')
# 
celeba_df['young'] = label_df['Young'].apply(lambda x: 'Y' if x== 1 else 'N')
# 
celeba_df['bangs'] = label_df['Bangs'].apply(lambda x: 'Y' if x== 1 else 'N')

celeba_df.to_csv('../public/assets/results_celeba.csv', index=False)
# %%
