import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import dataset
from sklearn.model_selection import StratifiedKFold


class GallbladderDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, train_mode=True):

        self.frame = pd.read_csv(csv_file, encoding='utf-8', header=None)
        '''
            self.frame:  image_name , label, patient_name
        '''
        self.root_dir = root_dir
        print('TestImage_DIR:',self.root_dir)
        self.transform = transform
        self.train_mode = train_mode

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if self.train_mode:
            '''
            Train Mode:getitem
            '''
            image_name = self.frame.iloc[idx,0]
            img,extension = os.path.splitext(image_name)
            folder_path = os.path.join(self.root_dir,self.get_folder(img))
            img_path = os.path.join(folder_path, image_name)
            img_name = self.frame.iloc[idx,2]
            image = self.image_loader(img_path, extension)
            label = int(self.frame.iloc[idx, 1])
            if self.transform is not None:
                image = self.transform(image)
            sample = {'image': image, 'label': label, 'img_name': img_name}
            return sample

        else:
            image_name = self.frame.iloc[idx, 0]
            img_path = os.path.join(self.root_dir, image_name)
            img_name = self.frame.iloc[idx, 2]
            _, extension = os.path.splitext(self.frame.iloc[idx, 0])
            image = self.image_loader(img_path, extension)
            label = int(self.frame.iloc[idx, 1])
            if self.transform is not None:
                image = self.transform(image)
            sample = {'image': image, 'label': label, 'img_name': img_name}
            return sample

    def image_loader(self, img_name, extension):
        if extension == '.JPG':
            # print('读取jpg')
            return self.read_jpg(img_name)
        elif extension == '.jpg':
            # print('读取jpg')
            return self.read_jpg(img_name)
        elif extension == '.DCM':
            # print('读取dcm')
            return self.read_dcm(img_name)
        elif extension == '.dcm':
            # print('读取dcm')
            return self.read_dcm(img_name)
        elif extension == '.Bmp':
            # print('读取Bmp')
            return self.read_bmp(img_name)
        elif extension == '.png':
            return self.read_png(img_name)
        elif extension == '.PNG':
            return self.read_png(img_name)

    def read_jpg(self, img_name):
        return Image.open(img_name).convert('L') # .JPG was transformed to gray-scale image

    def read_dcm(self, img_name):
        ds = sitk.ReadImage(img_name)
        img_array = sitk.GetArrayFromImage(ds)
        img_bitmap = Image.fromarray(img_array[0])
        return img_bitmap

    def read_bmp(self, img_name):
        return Image.open(img_name)

    def read_png(self, img_name):
        return Image.open(img_name).convert('L') # .PNG was transformed to gray-scale image


    def get_folder(self,img_name):
        '''
        Args:
            img_name:

        Returns:
            Training data folder of doctor
        '''
        if img_name.find('A') >= 0:
            return "docA/1/C_cropResult"
        elif img_name.find('B') >= 0:
            return "docB/1/C_cropResult"
        elif img_name.find('C') >= 0:
            return "docC/1/C_cropResult"
        elif img_name.find('D') >= 0:
            return "docD/1/C_cropResult"
        elif img_name.find('E') >= 0:
            return "docE/1/C_cropResult"
        elif img_name.find('F') >= 0:
            return "docF/1/C_cropResult"
        elif img_name.find('G') >= 0:
            return "docG/1/C_cropResult"
        else:
            return "OriginalData/crop_train_dataset"


