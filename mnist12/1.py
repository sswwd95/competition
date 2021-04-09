import os
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import albumentations as A


labels_df = pd.read_csv('../dacon12/dirty_mnist_2nd_answer.csv')[:]
imgs_dir = np.array(sorted(os.listdir('../dacon12/dirty_mnist_2nd/')))[:]
labels = np.array(labels_df.values[:,1:])
test_imgs_dir = np.array(sorted(os.listdir('../dacon12/test_dirty_mnist_2nd/')))

imgs=[]
for path in tqdm(imgs_dir[:]):
    img=cv2.imread(path, cv2.IMREAD_COLOR)
    imgs.append(img)
imgs=np.array(imgs)




DIR = '../dacon12/dirty_mnist_2nd/'
df_data = pd.read_csv(os.path.join(DIR,'../dacon12/dirty_mnist_2nd_answer.csv'))

row = df_data.iloc[8]
image_path = os.path.join(DIR, "train", row["StudyInstanceUID"] + ".jpg")
chosen_image = cv2.imread(image_path)

class MnistDataset_v1(Dataset):
    def __init__(self, imgs_dir=None, labels=None, transform=None, train=True):
        self.imgs_dir = imgs_dir
        self.labels = labels
        self.transform = transform
        self.train = train
        pass
    
    def __len__(self):
        # 데이터 총 샘플 수
        return len(self.imgs)
    
    def __getitem__(self, idx):
        # 1개 샘플 get
        img = cv2.imread(self.imgs_dir[idx], cv2.IMREAD_COLOR)
        img = self.transform(img)
        if self.train==True:
            label = self.labels[idx]
            return img, label
        else:
            return img
        
        pass
