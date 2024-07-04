import pandas as pd # type: ignore
import os
from torch.utils.data import Dataset
from torchvision.io import read_image


#
#   This is an extension of Dataset module from PyTorch.
#   We define our dataset like this:
#   - annotations_file = The path to the file with the annotations, in .csv format, with 2 comulmns (id and class label).
#   - img_dir = The path to the directory with all the images.
#   - transform = A single, or a composition of transformation to apply on the items of the dataset (Data Augumentation).
#   - target_transform = A single, or a composition of transformation to apply to the labels.
#
class ImageNet_Challenge_Dataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    #
    #   With this function we make this class iterable, yielding items everytime this is called.
    #   Everytime __getitem__ is called we retrieve the image and corresponding label with the given idx.
    #   We apply a normalization to the image, by diving for 255 [0, 255] -> [0, 1], for each channel.
    #   If defined, we apply transformations to images and labels.
    #
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0]) + '.jpg')
        image = read_image(img_path) / 255
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label