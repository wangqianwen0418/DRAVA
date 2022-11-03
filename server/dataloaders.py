from torch.utils.data import Dataset
import zarr
import torch
import pandas as pd
import os
import numpy as np
from PIL import Image

class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor, labels=None):
        self.data_tensor = data_tensor
        self.labels = labels

    def __getitem__(self, index):

        if torch.is_tensor(self.labels):
            return self.data_tensor[index], self.labels[index]
        return self.data_tensor[index], 0 # 0 as a dummy label

    def __len__(self):
        return self.data_tensor.size(0)

class CodeX_Dataset(Dataset):
    def __init__(self, root, transform=None, norm_method=None, split='train', in_channels=None, item_number=0):
        df = pd.read_csv(os.path.join(
            root, 'reg1_stitched_expressions.ome.tiff-cell_cluster.csv'))
        
        if item_number != 0:
            df = df.head(item_number)
        elif split != 'train':
            # use the first 1000 rows for validating and testing
            df = df.head(1000)
        self.img_dir = root
        self.img_names = df

        if norm_method:
            self.cell_patches = zarr.open(os.path.join(
                root, f'cell_patches_{norm_method}.zarr'), mode='r')
        else:
            self.cell_patches = zarr.open(os.path.join(
                root, 'cell_patches.zarr'), mode='r')
        if in_channels:
            self.cell_patches = self.cell_patches[:, :in_channels, :, : ]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # img_path = os.path.join(
        #     self.img_dir, 'cells', f'cell_{self.img_names.iloc[idx, 0]}.npy')
        # image = np.load(img_path)
        cell_id = self.img_names.iloc[idx, 0]
        image = np.array(self.cell_patches[cell_id])
        label = torch.tensor(self.img_names.iloc[idx, 1:])
        if self.transform:
            image = self.transform(image)
        return image, label

class CodeX_Grid_Dataset(Dataset):
    def __init__(self, root, transform=None, split='train', item_number=0):

        self.cell_grids = zarr.open(f'{root}.zarr', mode='r')
        if item_number != 0:
            self.cell_grids = self.cell_grids[:item_number]
        elif split != 'train':
            # use the first 1000 rows for validating and testing
            self.cell_grids = self.cell_grids[:1000]
        self.transform = transform

    def __len__(self):
        return len(self.cell_grids)

    def __getitem__(self, idx):
       
        image = np.array(self.cell_grids[idx])
        label = 0 # 0 as a dumpy label [TODO: get labels for cells in this grid]
        if self.transform:
            image = self.transform(image)
        return image, label

class CodeX_Landmark_Dataset(Dataset):
    def __init__(self, root, transform=None, num_cluster=0, split='train', item_number=0):
        df = pd.read_csv(os.path.join(
            root, 'reg1_stitched_expressions.ome.tiff-cell_cluster.csv'))
        
        if item_number != 0:
            df = df.head(item_number)
        elif split != 'train':
            # use the first 1000 rows for validating and testing
            df = df.head(1000)
        self.img_dir = root
        self.img_names = df

        
        self.cell_patches = zarr.open(os.path.join(
            root, f'cell_patches_{num_cluster}cluster.zarr'), mode='r')[:len(df)+1]

        self.num_cluster = num_cluster
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):

        cell_id = self.img_names.iloc[idx, 0]
        image = np.array(self.cell_patches[cell_id])
        pixels = image.flatten()
        # convert to one hot vector
        new_pixels = np.zeros( pixels.shape + (self.num_cluster+1,), dtype='int')
        new_pixels[np.arange(pixels.size), pixels] = 1
        new_image = new_pixels.reshape(image.shape+(self.num_cluster +1,))
        new_image = np.moveaxis(new_image, -1 , 0)

        label = torch.tensor(self.img_names.iloc[idx, 1:])
        if self.transform:
            new_image = self.transform(new_image)
        return new_image, label

class HiC_Dataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, chr=None):
        df = pd.read_csv(os.path.join(root, 'label.csv'))
        if (chr != None):
            # e.g., chr = 'chr5'
            self.img_labels = df[df['img'].str.contains(chr)]
        else:
            self.img_labels = df
        self.img_dir = root
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir, f'{self.img_labels.iloc[idx, 0]}.jpg')
        image = Image.open(img_path).convert('L')
        #
        label = self.img_labels.iloc[idx].copy()
        try:
            # get the CHR number from the jpg name
            label[0] = float(label[0].split(':')[0].replace('chr', ''))
        except Exception:
            label[0] = 7  # chr 7 dataset
        label = label.to_numpy(dtype='float')

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class IDC_Dataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, chr=None):
        df = pd.read_csv(os.path.join(root, 'label.csv'))
        self.img_labels = df
        self.img_dir = root
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir, f'{self.img_labels.iloc[idx, 0]}')
        image = Image.open(img_path)
        #
        label = self.img_labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

