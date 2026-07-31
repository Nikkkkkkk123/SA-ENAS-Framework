from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch

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
            transforms.Resize((self._image_size, self._image_size)),
            transforms.ToTensor(),
        ])


        fullDataset = datasets.ImageFolder(root=path, transform=transform)

        trainSize = int(0.8 * len(fullDataset))
        valSize = int(0.1 * len(fullDataset))
        testSize = len(fullDataset) - trainSize - valSize

        train_data, val_data, test_data= torch.utils.data.random_split(
            fullDataset, 
            [trainSize, valSize, testSize],
            generator=torch.Generator().manual_seed(42)
        )

        
        trainLoader = DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=1)
        valLoader = DataLoader(val_data, batch_size=bs, shuffle=False, num_workers=1)
        testLoader = DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=1)

        return trainLoader, valLoader, testLoader, fullDataset.classes
