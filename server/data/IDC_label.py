#%%
from glob import glob
import pandas as pd

#%%
imagePatches = glob('./IDC_regular_ps50_idx5/**/*.png', recursive=True)

import fnmatch
patternZero = '*class0.png'
patternOne = '*class1.png'
classZero = fnmatch.filter(imagePatches, patternZero)
classOne = fnmatch.filter(imagePatches, patternOne)

y = []
for img in imagePatches:
    if img in classZero:
        y.append(0)
    elif img in classOne:
        y.append(1)

images_df = pd.DataFrame()

images_df["images"] = imagePatches
images_df["images"] = images_df["images"].apply(lambda x:x.replace('./IDC_regular_ps50_idx5/', ''))
images_df["labels"] = y
# %%
images_df.to_csv('./IDC_regular_ps50_idx5/label.csv')
# %%
