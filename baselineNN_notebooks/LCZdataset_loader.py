from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from skimage import io#, transform
import numpy as np

class LCZdataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.landmarks_frame.iloc[idx, 0]
        image = io.imread(img_name)
        classLabel = self.landmarks_frame.iloc[idx, 1]
        classLabel = np.array([classLabel])
        #landmarks = landmarks.astype('float').reshape(-1, 2)

        #print(image.shape)
        image = image[:,:,[1,2,3,4,5,6,7,10,11,12]]/10000.0

        sample = {'image': image, 'label': classLabel}

        if self.transform:
            sample = self.transform(sample)

        return sample['image'], sample['label']-1#sample

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        #landmarks = landmarks - [left, top]

        return {'image': image, 'label': label}


class ToTensor(object): 
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image.astype("float")).float(),#torch.from_numpy(image.astype("float")).float()
                'label': torch.from_numpy(label)}
