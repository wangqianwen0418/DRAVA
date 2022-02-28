#%%
from glob import glob
from operator import index
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

images_df["image"] = imagePatches
images_df["image"] = images_df["images"].apply(lambda x:x.replace('./IDC_regular_ps50_idx5/', ''))
images_df["label"] = y
# %%
images_df.to_csv('./IDC_regular_ps50_idx5/label.csv')
# %%
# check image size
from PIL import Image
num_wrong_size = 0
square_imgs = []
square_y = []
for img_path in imagePatches:
    image = Image.open(img_path)
    width, height = image.size
    if not  (width == 50 and height == 50):
        print('width:', width, 'height:', height, img_path)
        num_wrong_size += 1
    else:
        square_imgs.append(img_path.replace('./IDC_regular_ps50_idx5/', ''))
        square_y.append(0 if img in imagePatches else 1)
#%%
square_img_df = pd.DataFrame()
square_img_df['image'] = square_imgs
square_img_df['label'] = square_y
square_img_df.to_csv('./IDC_regular_ps50_idx5/label_square.csv', index=False)
# %%
# add image path to the result file
result_file = '../../front/public/assets/results_IDC_all.csv'
label_file = './IDC_regular_ps50_idx5/label.csv'
patient_id = '16166'

result_df = pd.read_csv(result_file)
label_df = pd.read_csv(label_file)
patient_df = pd.DataFrame()

result_df['img_path'] = label_df['image']
result_df['label'] = label_df['label']
result_df = result_df[result_df['img_path'].str.startswith(f'{patient_id}/')]
result_df = result_df.sample(frac=1) # shuffle rows

result_df.to_csv('../../front/public/assets/results_IDC.csv', index=False)
# %%
