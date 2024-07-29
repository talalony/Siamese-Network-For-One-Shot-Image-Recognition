import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseVerifier(nn.Module):
    def __init__(self, channels=1, image_size=105):
        super(SiameseVerifier, self).__init__()

        self.twin = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=10, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        image_size = (((image_size - 9) / 2 - 6) / 2 - 3) / 2 - 3
        if image_size % 1 != 0:
            raise ValueError("Image size does not fit")
        else:
            image_size = int(image_size)

        self.fc1 = nn.Linear(256 * image_size * image_size, 4096)
        self.fc2 = nn.Linear(4096, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.5, std=1e-2)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=2e-1)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.5, std=1e-2)

    def forward(self, input1, input2, training=False):
        output1 = self.twin(input1)
        output2 = self.twin(input2)
        output1 = F.sigmoid(self.fc1(output1))
        output2 = F.sigmoid(self.fc1(output2))
        output = torch.abs(output1 - output2)
        output = self.fc2(output)
        if training:
            return output
        return F.sigmoid(output)
