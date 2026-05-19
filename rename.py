from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision import datasets
class DataLoaders:
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        return images, labels
    def load_data(dataset_name='mnist'):
        traindataset = load_dataset('mnist', split='train', streaming=True)

        transform = v2.Compose([
            v2.Resize((28, 28)),
            v2.Grayscale(num_output_channels=1),
            v2.ToTensor(),
            v2.Normalize((0.1307,), (0.3081,))
        ])
        newTrainSet = []
        for item in traindataset:
            image = transform(item['image'])
            label = item['label']
            newTrainSet.append((image, label))
        # Convert to DataLoader
        trainLoader = DataLoader(newTrainSet, batch_size=2, shuffle=True, collate_fn=DataLoaders.collate_fn)
        classes = traindataset.features['label'].names
        return trainLoader, trainLoader, classes


