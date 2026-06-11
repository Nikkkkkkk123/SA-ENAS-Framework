from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision import datasets
import os
class DataLoaders:
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        return images, labels
    def load_data(dataset_name='mnist'):

        transform = v2.Compose([
            v2.Resize((28, 28)),
            v2.Grayscale(num_output_channels=1),
            v2.ToTensor(),
            v2.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load the local datasets
        trainData = datasets.ImageFolder (
            root='/run/media/nikk/HI/TrainImg',
            transform=transform,
        )
        trainLoader = DataLoader(trainData, batch_size=32, shuffle=False)
        classes = trainData.classes

        return trainLoader, trainLoader, classes


