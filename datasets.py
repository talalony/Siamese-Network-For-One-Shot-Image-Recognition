import os
import random

import torch
from torch.utils.data import Dataset
from PIL import Image


class OmniglotTrainDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super(OmniglotTrainDataset, self).__init__()

        self.data, self.num_of_classes, self.size = self.load_data(data_path)
        self.transform = transform

    @staticmethod
    def load_data(data_path: str):
        data = {}
        idx = 0
        size = 0
        for alpha in os.listdir(data_path):
            if alpha.startswith("."):
                continue
            for char in os.listdir(os.path.join(data_path, alpha)):
                if char.startswith("."):
                    continue
                data[idx] = []
                for file in os.listdir(os.path.join(data_path, alpha, char)):
                    if file.startswith("."):
                        continue
                    if file.endswith(".jpg") or file.endswith(".png"):
                        file_path = os.path.join(data_path, alpha, char, file)
                        try:
                            data[idx].append(Image.open(file_path).convert('L'))
                        except:
                            continue
                        size += 1
                if not data[idx]:
                    del data[idx]
                    continue
                idx += 1

        return data, idx, size

    def __len__(self):
        return self.size*8

    def __getitem__(self, idx):
        if idx % 2 == 0:
            label = True
            rand_idx = random.randint(0, self.num_of_classes - 1)
            image1 = random.choice(self.data[rand_idx])
            image2 = random.choice(self.data[rand_idx])
        else:
            label = False
            rand_idx = random.randint(0, self.num_of_classes - 1)
            image1 = random.choice(self.data[rand_idx])
            rand_idx2 = random.randint(0, self.num_of_classes - 1)
            while rand_idx2 == rand_idx:
                rand_idx2 = random.randint(0, self.num_of_classes - 1)
            image2 = random.choice(self.data[rand_idx2])

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, torch.tensor([label]).float()


class OmniglotTestDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super(OmniglotTestDataset, self).__init__()

        self.data, self.num_of_classes, self.size = self.load_data(data_path)
        self.transform = transform

    @staticmethod
    def load_data(data_path: str):
        data = {}
        idx = 0
        size = 0
        for alpha in os.listdir(data_path):
            if alpha.startswith("."):
                continue
            for char in os.listdir(os.path.join(data_path, alpha)):
                if char.startswith("."):
                    continue
                data[idx] = []
                for file in os.listdir(os.path.join(data_path, alpha, char)):
                    if file.startswith("."):
                        continue
                    if file.endswith(".jpg") or file.endswith(".png"):
                        file_path = os.path.join(data_path, alpha, char, file)
                        try:
                            data[idx].append(Image.open(file_path).convert('L'))
                        except:
                            continue
                        size += 1
                if not data[idx]:
                    del data[idx]
                    continue
                idx += 1

        return data, idx, size

    def __len__(self):
        return self.size*2

    def __getitem__(self, idx):
        if idx % 2 == 0:
            label = True
            rand_idx = random.randint(0, self.num_of_classes - 1)
            image1 = random.choice(self.data[rand_idx])
            image2 = random.choice(self.data[rand_idx])
        else:
            label = False
            rand_idx = random.randint(0, self.num_of_classes - 1)
            image1 = random.choice(self.data[rand_idx])
            rand_idx2 = random.randint(0, self.num_of_classes - 1)
            while rand_idx2 == rand_idx:
                rand_idx2 = random.randint(0, self.num_of_classes - 1)
            image2 = random.choice(self.data[rand_idx2])

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, torch.tensor([label]).float()
