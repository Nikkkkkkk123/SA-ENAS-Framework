from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch
import os

class ByteDataset (Dataset):
    def __init__ (self, filePath, label = None):
        self.filePath = filePath
        self.label = label
        self.classes = []
        for root, dirs, files in os.walk(filePath):
            self.samples = [(os.path.join(file.split(".")[0]), None) for file in files if file.endswith('.asm')]

        firstFilePath = self.samples[0][0]
        # print fil ename

        for dir in os.listdir("D:\\TrainImg"):
            self.classes.append(dir)

    def __len__ (self):
        return len(self.samples)

    def __getitem__(self, key):
        sample = self.samples[key]
        image = None
        label = None
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            # transforms.Resize((self._image_size, self._image_size),
            #                   interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        for dir in os.listdir("D:\\TrainImg"):
            if os.path.isfile(os.path.join("D:\\TrainImg", dir, os.path.basename(sample[0]) + ".bytes.png")):
                label = dir
                image = Image.open(os.path.join("D:\\TrainImg", dir, os.path.basename(sample[0]) + ".bytes.png"))
                image = transform(image)
                break

        return image, self.classes.index(label) if label is not None else -1

class Dataset:
    _batch_size: int
    _image_size: int
    _input_channels: int

    def __init__ (self):
        return None

    def getDataset(self, batchsize: int, image_size: int, input_channels: int):
        bs = batchsize
        self._image_size = image_size
        self._input_channels = input_channels

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=self._input_channels),
            # transforms.Resize((self._image_size, self._image_size),
            #                   interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        path = "D:\\train"

        fullDataset = ByteDataset(filePath=path)

        trainSize = int(0.8 * len(fullDataset))
        valSize = int(0.1 * len(fullDataset))
        testSize = len(fullDataset) - trainSize - valSize

        train_data, val_data, test_data= torch.utils.data.random_split(
            fullDataset, 
            [trainSize, valSize, testSize],
            generator=torch.Generator().manual_seed(42)
        )

        
        trainLoader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=4, persistent_workers=True)
        valLoader = DataLoader(val_data, batch_size=bs, shuffle=False, num_workers=4, persistent_workers=True)
        testLoader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=4, persistent_workers=True)

        return trainLoader, valLoader, testLoader, fullDataset.classes
