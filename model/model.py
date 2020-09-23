import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

import torch

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

"for LCZ"
def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        #nn.MaxPool2d(2)
    )

class Net_LCZ(nn.Module):
    def __init__(self):
        super(Net_LCZ, self).__init__()
        self.conv1 = conv3x3(10, 16)
        self.conv2 = conv3x3(16, 32)
        self.conv3 = conv3x3(32, 64)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)

        self.conv4 = conv3x3(64, 128)
        self.conv5 = conv3x3(128, 256)
        self.conv6 = conv3x3(256, 256)

        self.conv7 = nn.Conv2d(64, 64, 5)
        self.conv8 = nn.Conv2d(64, 64, 7)
        self.conv9 = nn.Conv2d(64, 64, 13)

        #self.fc1 = nn.Linear(32 * 13 * 13, 120)
        #self.fc2 = nn.Linear(120, 84)

        self.pool3 = nn.AdaptiveAvgPool2d(1)
        self.pool4 = nn.AdaptiveMaxPool2d(1)

        self.fc3 = nn.Linear(64 * 5, 17)

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        #x1 = self.pool1(x0)
        #x2 = self.pool2(x0)
        #x = torch.cat([x1, x2], dim=1)
        #print(x.shape)

        #x = self.conv4(x)
        #x = self.conv5(x)
        #x = self.conv6(x)

        x5 = self.conv7(x)
        x5 = self.pool4(x5)

        x6 = self.conv8(x)
        x6= self.pool4(x6)

        x7 = self.conv9(x)
        x7= self.pool4(x7)

        x3 = self.pool3(x)
        x4 = self.pool4(x)

        x = torch.cat([x3, x4, x5, x6, x7], dim=1)

        x = x.view(-1, 64 * 5)
        x = self.fc3(x)

        return x
