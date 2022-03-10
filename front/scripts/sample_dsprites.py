#%%
from math import floor
import numpy as np

import pandas as pd
import math

raw_data = np.load('../../server/data/dsprites.npz', encoding='bytes')
all_results = pd.read_csv('../public/assets/results_dsprites_all.csv')

# %%
sample_num =1000
ratio = math.floor(len(raw_data)/sample_num)
sample_array = [i*ratio for i in range(sample_num)]


import random
random.shuffle(sample_array)
# %%
test_data_img = raw_data['imgs'][sample_array]
test_data_value = raw_data['latents_values'][sample_array]
test_data_class = raw_data['latents_classes'][sample_array]
results = all_results.iloc[sample_array]
#%%
np.savez('../../server/data/dsprites_test.npz', imgs=test_data_img, latents_values=test_data_value, latents_classes=test_data_class)
results.to_csv('../public/assets/results_dsprites.csv', index=False)
# %%
