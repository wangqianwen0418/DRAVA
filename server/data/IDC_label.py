#%%
from glob import glob
from operator import index
import pandas as pd
from PIL import Image
import fnmatch

#%%
def generate_IDC_label():
    imagePatches = glob('./IDC_regular_ps50_idx5/**/*.png', recursive=True)


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
    images_df["image"] = images_df["image"].apply(lambda x:x.replace('./IDC_regular_ps50_idx5/', ''))
    images_df["label"] = y
    # %%
    images_df.to_csv('./IDC_regular_ps50_idx5/label_all.csv', index=False)
    # %%
    # check image size

    num_wrong_size = 0
    square_imgs = []
    square_y = []
    for img_path in imagePatches:
        image = Image.open(img_path)
        width, height = image.size
        if not  (width == 50 and height == 50):
            # print('width:', width, 'height:', height, img_path)
            num_wrong_size += 1
        else:
            square_imgs.append(img_path.replace('./IDC_regular_ps50_idx5/', ''))
            square_y.append(0 if img_path in classZero else 1)
    #%%
    square_img_df = pd.DataFrame()
    square_img_df['image'] = square_imgs
    square_img_df['label'] = square_y
    square_img_df.to_csv('./IDC_regular_ps50_idx5/label.csv', index=False)

# %%
# add image path to the result file
####################################
def add_info2results():
    result_file = '../../front/public/assets/results_IDC_all.csv'
    label_file = './IDC_regular_ps50_idx5/label.csv'
    pred_file = '../IDC_results.csv'

    patient_id = '12749'

    result_df = pd.read_csv(result_file)
    label_df = pd.read_csv(label_file)
    pred_df = pd.read_csv(pred_file)

    result_df['img_path'] = label_df['image']
    result_df['label'] = label_df['label']
    result_df = result_df[result_df['img_path'].str.startswith(f'{patient_id}/')]

    sub_pred = pred_df[pred_df['image'].str.startswith(f'{patient_id}/')]

    new_df = result_df.merge(sub_pred, left_on ='img_path', right_on = 'image' )
    new_df = result_df.sample(frac=1) # shuffle rows

    new_df.to_csv('../../front/public/assets/results_IDC_test.csv', index=False)

def process_pred_file():
    pred_file = '../IDC_results.csv'
    label_file = './IDC_regular_ps50_idx5/label.csv'
    pred_df = pd.read_csv(pred_file)
    label_df = pd.read_csv(label_file)

    sub_label = label_df.iloc[202154:, :].reset_index(drop=True)


    pred_df['label'] = sub_label['label']
    pred_df['image'] = sub_label['image']
    pred_df['confidence'] = (pred_df['score'] - 0.5).abs() * 2
    pred_df['acc'] = (pred_df['score'] - pred_df['label']).abs() < 0.5

    pred_df.to_csv('./IDC_pred.csv', index=False)
# %%
if __name__=='__main__':
    # generate_IDC_label()
    add_info2results()