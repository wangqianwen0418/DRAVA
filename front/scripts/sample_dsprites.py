#%%
from math import floor
import numpy as np

import pandas as pd
import math

raw_data = np.load('../../server/data/dsprites.npz', encoding='bytes')['imgs']
all_results = pd.read_csv('../public/assets/results_dsprites_all.csv')

# %%
sample_num =1000
ratio = math.floor(len(raw_data)/sample_num)
sample_array = [i*ratio for i in range(sample_num)]


import random
random.shuffle(sample_array)
# %%
test_data = raw_data[sample_array]
results = all_results.iloc[sample_array]
#%%
np.save('../../server/data/dsprites_test.npy', test_data)
results.to_csv('../public/assets/results_dsprites.csv', index=False)
# %%
