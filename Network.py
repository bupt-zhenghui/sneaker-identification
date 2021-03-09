import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256),

        )

        self.linear_layers = nn.Sequential(
            nn.Linear(10 * 10 * 256, 1024),
            nn.Linear(1024, 4)
        )

    def forward(self, x):
        conv_out = self.conv(x)
        linear_input = conv_out.view(conv_out.size(0), -1)
        linear_out = self.linear_layers(linear_input)
        return linear_out


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.conv = nn.Sequential(             # (224, 224)
            nn.Conv2d(3, 64, 3, 1, 1),  # 1
            nn.Conv2d(64, 64, 3, 1, 1),  # 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64),                # (112, 112)

            nn.Conv2d(64, 128, 3, 1, 1),  # 3
            nn.Conv2d(128, 128, 3, 1, 1),  # 4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(128),               # (56, 56)

            nn.Conv2d(128, 256, 3, 1, 1),  # 5
            nn.Conv2d(256, 256, 3, 1, 1),  # 6
            nn.Conv2d(256, 256, 3, 1, 1),  # 7
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256),               # (28, 28)

            nn.Conv2d(256, 512, 3, 1, 1),  # 8
            nn.Conv2d(512, 512, 3, 1, 1),  # 9
            nn.Conv2d(512, 512, 3, 1, 1),  # 10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(512),               # (14, 14)

            nn.Conv2d(512, 512, 3, 1, 1),  # 11
            nn.Conv2d(512, 512, 3, 1, 1),  # 12
            nn.Conv2d(512, 512, 3, 1, 1),  # 13
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),       # (7, 7)
        )

        self.linear = nn.Sequential(
            nn.Linear(7 * 7 * 512, 64),
            nn.Linear(64, 2)                   # (4096, 1000)
        )

    def forward(self, x):
        conv_output = self.conv(x)
        linear_input = conv_output.view(conv_output.size(0), -1)
        output = self.linear(linear_input)
        return output

