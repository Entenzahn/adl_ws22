import torch
import torch.nn as nn
import torch.nn.functional as F


class SplitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop_rate = 0.5

        self.drop_layer = nn.Dropout2d(p=self.drop_rate)

        self.stem_conv1 = nn.Conv2d(1, 6, 3, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.stem_bn1 = nn.BatchNorm2d(6)
        self.stem_conv2 = nn.Conv2d(6, 6, 3)
        self.stem_bn2 = nn.BatchNorm2d(6)
        self.stem_conv3 = nn.Conv2d(6, 12, 3)
        self.stem_bn3 = nn.BatchNorm2d(12)
        self.stem_conv4 = nn.Conv2d(12, 19, 3)
        self.stem_bn4 = nn.BatchNorm2d(19)
        self.stem_conv5 = nn.Conv2d(19, 38, 3)
        self.stem_bn5 = nn.BatchNorm2d(38)

        self.tonic_conv1 = nn.Conv2d(38, 12, 1)
        self.tonic_bn1 = nn.BatchNorm2d(12)
        self.tonic_conv2 = nn.Conv2d(12, 24, 1)
        self.tonic_bn2 = nn.BatchNorm2d(24)
        self.tonic_conv3 = nn.Conv2d(24, 12, 1)
        self.tonic_bn3 = nn.BatchNorm2d(12)
        self.tonic_conv4 = nn.Conv2d(12, 24, 1)
        self.tonic_bn4 = nn.BatchNorm2d(24)

        self.tonic_fc1 = nn.Linear(203472, 3200)
        self.tonic_fc2 = nn.Linear(3200, 12)

        self.mode_conv1 = nn.Conv2d(38, 6, 3)
        self.mode_bn1 = nn.BatchNorm2d(6)
        self.mode_conv2 = nn.Conv2d(6, 9, 5)
        self.mode_bn2 = nn.BatchNorm2d(9)
        self.mode_conv3 = nn.Conv2d(9, 12, 3)
        self.mode_bn3 = nn.BatchNorm2d(12)

        self.mode_fc1 = nn.Linear(69768, 24)
        self.mode_fc2 = nn.Linear(24, 1)

        self.out_layer = nn.Linear(13, 24)

    def forward(self, x):
        # print(x.device)
        # print(self.conv1.state_dict()['bias'].device)
        x = self.stem_bn1(F.relu(self.stem_conv1(x)))
        x = self.stem_bn2(F.relu(self.stem_conv2(x)))
        x = self.stem_bn3(F.relu(self.stem_conv3(x)))
        x = self.stem_bn4(F.relu(self.stem_conv4(x)))
        x = self.stem_bn5(F.relu(self.stem_conv5(x)))
        x = self.drop_layer(x)

        x_tonic = self.tonic_bn1(F.relu(self.tonic_conv1(x)))
        x_tonic = self.tonic_bn2(F.relu(self.tonic_conv2(x_tonic)))
        x_tonic = self.tonic_bn3(F.relu(self.tonic_conv3(x_tonic)))
        x_tonic = self.tonic_bn4(F.relu(self.tonic_conv4(x_tonic)))
        x_tonic = self.drop_layer(x_tonic)
        x_tonic = torch.flatten(x_tonic, 1)  # flatten all dimensions except batch
        x_tonic = self.tonic_fc1(x_tonic)
        x_tonic = self.tonic_fc2(x_tonic)

        x_mode = self.mode_bn1(F.relu(self.mode_conv1(x)))
        x_mode = self.mode_bn2(F.relu(self.mode_conv2(x_mode)))
        x_mode = self.mode_bn3(F.relu(self.mode_conv3(x_mode)))
        x_mode = self.drop_layer(x_mode)
        x_mode = torch.flatten(x_mode, 1)  # flatten all dimensions except batch
        x_mode = self.mode_fc1(x_mode)
        x_mode = self.mode_fc2(x_mode)

        """x_left = F.relu(self.block_b_conv_left1(x))        
        x_mid = self.block_b_bn_mid1(F.relu(self.block_b_conv_mid1(x)))
        x_mid = self.block_b_bn_mid2(F.relu(self.block_b_conv_mid2(x_mid)))
        x_mid = F.relu(self.block_b_conv_mid3(x_mid))
        x = self.block_b_bn_cat(torch.cat([x_left,x_mid],1))
        x = self.drop_layer(x)"""

        x = torch.cat([x_tonic, x_mode], 1)
        x = F.softmax(self.out_layer(x), dim=1)

        return x


class ColNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop_rate = 0.3

        self.drop_layer = nn.Dropout2d(p=self.drop_rate)

        self.conv1 = nn.Conv2d(1, 12, (14,1), stride=(14,1))
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, 3)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=3)
        self.bn3 = nn.BatchNorm2d(24)
        self.out_layer1 = nn.Linear(15408, 2400)
        self.out_layer2 = nn.Linear(2400, 24)

    def forward(self, x):
        # print(x.device)
        # print(self.conv1.state_dict()['bias'].device)
        #print(f"Original dimensions: {x.shape}")
        x = self.bn1(F.relu(self.conv1(x)))
        #print(f"First layer: {x.shape}")
        x = self.bn2(F.relu(self.conv2(x)))
        #print(f"Second layer: {x.shape}")
        x = self.bn3(F.relu(self.conv3(x)))
        #print(f"Third layer: {x.shape}")
        x = self.drop_layer(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.out_layer1(x))
        #print(f"Fully conncected layer: {x.shape}")
        x = F.softmax(self.out_layer2(x), dim=1)
        #print(f"Done lol")

        return x
