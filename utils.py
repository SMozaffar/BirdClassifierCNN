from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import csv
import os
import torch


def get_labels(csv_path):
    label_names = set()
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            label_names.add(row['labels'])

    return sorted(label_names)


# Get label names
global LABEL_NAMES 
global num_classes
LABEL_NAMES = get_labels('/content/BirdClassifierCNN/data/birds.csv')
num_classes = len(LABEL_NAMES)


class BirdsDataset(Dataset):
    def __init__(self, dataset_path, split='train'):
        self.data = []
        self.labels = []
        label_path = os.path.join(dataset_path, 'birds.csv')
        
        # Read csv file
        with open(label_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['filepaths'].split('/')[0] != split:
                    continue
                else:
                    img_path = os.path.join(dataset_path, split, row['filepaths'].split('/')[-2], row['filepaths'].split('/')[-1])
                    label = LABEL_NAMES.index(row['labels'])
                    self.data.append(img_path)
                    self.labels.append(label)
        
        # Transforms PIL image to tensor
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        return: (img, label)
        """
        # Load the image and transform to tensor
        img = Image.open(self.data[idx])
        img = self.transform(img)
        label = self.labels[idx]
        return img, label


def load_data(dataset_path, num_workers=0, batch_size=128, **kwargs):
    dataset = BirdsDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
