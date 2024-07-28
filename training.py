import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from tqdm import tqdm

from datasets import OmniglotTrainDataset, OmniglotTestDataset


def train(data_path, model, device, num_epochs=200, batch_size=128, learning_rate=0.00006, save_length=50):
    data_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])

    dataset = OmniglotTrainDataset(data_path, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"training on {len(dataset)} same/different pairs")

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    losses = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

        for image1, image2, label in progress_bar:
            image1, image2, label = image1.to(device), image2.to(device), label.to(device)

            optimizer.zero_grad()

            output = model(image1, image2, training=True)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / len(dataloader))

        epoch_loss = total_loss / len(dataloader)
        losses.append(epoch_loss)

        if epoch != 0 and epoch % save_length == 0:
            torch.save(model.state_dict(), f"model_omniglot_epoch{epoch}.pth")

    with open("training_losses.json", "w") as f:
        json.dump(losses, f)


def test(data_path, model, device):
    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = OmniglotTestDataset(data_path, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"total length: {len(dataset)}")

    model.eval()
    same = 0
    for image1, image2, label in dataloader:
        image1, image2, label = image1.to(device), image2.to(device), label.to(device)
        output = model(image1, image2)

        pred = 0
        if output.item() > 0.5:
            pred = 1

        if pred == label.item():
            same += 1

    print(f"Accuracy: {same / len(dataset)}")
