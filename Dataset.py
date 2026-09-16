from pathlib import Path

from PIL import Image
from sympy import fu
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch
import os

"""
This below should not be needed like this. Interpolation layer is only required when generating new adversarial samples. So normal transformer can be utilised for the clean dataset
"""
# class ByteDataset (Dataset):
#     def __init__ (self, filePath, label = None):
#         self.filePath = filePath
#         self.label = label
#         self.classes = []
#         for root, dirs, files in os.walk(filePath):
#             self.samples = [(os.path.join(file.split(".")[0]), None) for file in files if file.endswith('.asm')]

#         firstFilePath = self.samples[0][0]
#         # print fil ename

#         for dir in os.listdir("D:\\TrainImg"):
#             self.classes.append(dir)

#     def __len__ (self):
#         return len(self.samples)

#     def __getitem__(self, key):
#         sample = self.samples[key]
#         image = None
#         label = None
#         transform = transforms.Compose([
#             transforms.Grayscale(num_output_channels=1),
#             # transforms.Resize((self._image_size, self._image_size),
#             #                   interpolation=transforms.InterpolationMode.BILINEAR),
#             transforms.ToTensor(),
#         ])
#         for dir in os.listdir("D:\\TrainImg"):
#             if os.path.isfile(os.path.join("D:\\TrainImg", dir, os.path.basename(sample[0]) + ".bytes.png")):
#                 label = dir
#                 image = Image.open(os.path.join("D:\\TrainImg", dir, os.path.basename(sample[0]) + ".bytes.png"))
#                 image = transform(image)
#                 break

#         return image, self.classes.index(label) if label is not None else -1

class Dataset:
    _batch_size: int
    _image_size: int
    _input_channels: int

    def __init__ (self):
        return None

    def getDataset(self, batchsize: int, image_size: int, input_channels: int, dataset_path: str = "D:\\"):
        bs = batchsize
        self._image_size = image_size
        self._input_channels = input_channels

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=self._input_channels),
            transforms.Resize((self._image_size, self._image_size)),
            transforms.ToTensor(),
        ])

        path = dataset_path
        malpath = os.path.join(path, "/TrainImg/")
        fullDataset = datasets.ImageFolder(root=malpath, transform=transform)
        benPath = os.path.join(path, "/TrainBenImg")
        fullBenDataset = datasets.ImageFolder(root=benPath, transform=transform) # Offset the labels of the benign dataset
        fullBenDataset.class_to_idx = {class_name: idx + len(fullDataset.classes) for class_name, idx in fullBenDataset.class_to_idx.items()}
        fullBenDataset.samples = [(sample[0], sample[1] + len(fullDataset.classes)) for sample in fullBenDataset.samples]
        fullBenDataset.targets = [target + len(fullDataset.classes) for target in fullBenDataset.targets]
        trainSize = int(0.8 * len(fullDataset))
        valSize = int(0.1 * len(fullDataset))
        testSize = len(fullDataset) - trainSize - valSize

        benTrainSize = int(0.8 * len(fullBenDataset))
        benValSize = int(0.1 * len(fullBenDataset))
        benTestSize = len(fullBenDataset) - benTrainSize - benValSize

        train_data, val_data, test_data= torch.utils.data.random_split(
            fullDataset, 
            [trainSize, valSize, testSize],
            generator=torch.Generator().manual_seed(42)
        )

        bentrain_data, benval_data, bentest_data= torch.utils.data.random_split(
            fullBenDataset, 
            [benTrainSize, benValSize, benTestSize],
            generator=torch.Generator().manual_seed(42)
        )

        fullTrain = torch.utils.data.ConcatDataset([train_data, bentrain_data])
        fullVal = torch.utils.data.ConcatDataset([val_data, benval_data])
        fullTest = torch.utils.data.ConcatDataset([test_data, bentest_data])

        # check issue with class confusion
        print("fullDataset classes:", fullDataset.classes, fullDataset.class_to_idx)
        print("fullBenDataset classes:", fullBenDataset.classes, fullBenDataset.class_to_idx)


        trainLoader = DataLoader(fullTrain, batch_size=bs, shuffle=True, num_workers=4, persistent_workers=True)
        valLoader = DataLoader(fullVal, batch_size=bs, shuffle=False, num_workers=4, persistent_workers=True)
        testLoader = DataLoader(fullTest, batch_size=bs, shuffle=False, num_workers=4, persistent_workers=True)

        classes = fullDataset.classes + fullBenDataset.classes

        return trainLoader, valLoader, testLoader, classes
