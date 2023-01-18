import torch
import torch.nn as nn
import torch.nn.functional as F


class KeyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop_rate = 0.5

        self.drop_layer = nn.Dropout2d(p=self.drop_rate)

        self.stem_conv1 = nn.Conv2d(1, 6, 3, stride=2)
        self.stem_bn1 = nn.BatchNorm2d(6)
        self.stem_conv2 = nn.Conv2d(6, 6, 3)
        self.stem_bn2 = nn.BatchNorm2d(6)
        self.stem_conv3 = nn.Conv2d(6, 12, 3)
        self.stem_bn3 = nn.BatchNorm2d(12)
        self.stem_conv4 = nn.Conv2d(12, 19, 3)
        self.stem_bn4 = nn.BatchNorm2d(19)
        self.stem_conv5 = nn.Conv2d(19, 38, 3)
        self.stem_bn5 = nn.BatchNorm2d(38)

        self.block_a18_conv_left1 = nn.Conv2d(38, 9, 1)
        self.block_a18_bn_left1 = nn.BatchNorm2d(9)
        self.block_a18_conv_left2 = nn.Conv2d(9, 12, 5)
        self.block_a18_conv_mid1 = nn.Conv2d(38, 12, 1)
        self.block_a18_bn_mid1 = nn.BatchNorm2d(12)
        self.block_a18_conv_mid2 = nn.Conv2d(12, 19, 1)
        self.block_a18_bn_mid2 = nn.BatchNorm2d(19)
        self.block_a18_conv_mid3 = nn.Conv2d(19, 19, 5)
        self.block_a18_conv_right1 = nn.Conv2d(38, 18, 5)
        self.block_a18_bn_cat = nn.BatchNorm2d(18 + 19 + 12)

        self.block_a24_conv_left1 = nn.Conv2d(18 + 19 + 12, 9, 1)
        self.block_a24_bn_left1 = nn.BatchNorm2d(9)
        self.block_a24_conv_left2 = nn.Conv2d(9, 12, 5)
        self.block_a24_conv_mid1 = nn.Conv2d(18 + 19 + 12, 12, 1)
        self.block_a24_bn_mid1 = nn.BatchNorm2d(12)
        self.block_a24_conv_mid2 = nn.Conv2d(12, 19, 1)
        self.block_a24_bn_mid2 = nn.BatchNorm2d(19)
        self.block_a24_conv_mid3 = nn.Conv2d(19, 19, 5)
        self.block_a24_conv_right1 = nn.Conv2d(18 + 19 + 12, 24, 5)
        self.block_a24_bn_cat = nn.BatchNorm2d(24 + 19 + 12)

        self.out_layer = nn.Linear(278080, 24)

        """self.block_b_conv_left1 = nn.Conv2d(24+19+12, 76, 3)
        self.block_b_conv_mid1 = nn.Conv2d(24+19+12, 12, 1)
        self.block_b_bn_mid1 = nn.BatchNorm2d(12)
        self.block_b_conv_mid2 = nn.Conv2d(12, 19, 3)
        self.block_b_bn_mid2 = nn.BatchNorm2d(19)
        self.block_b_conv_mid3 = nn.Conv2d(19, 19, 1)
        self.block_b_bn_cat = nn.BatchNorm2d(19+76)"""
        # self.out_layer = nn.Linear(453530, 24)

    def forward(self, x):
        # print(x.device)
        # print(self.conv1.state_dict()['bias'].device)
        x = self.stem_bn1(F.relu(self.stem_conv1(x)))
        x = self.stem_bn2(F.relu(self.stem_conv2(x)))
        x = self.stem_bn3(F.relu(self.stem_conv3(x)))
        x = self.stem_bn4(F.relu(self.stem_conv4(x)))
        x = self.stem_bn5(F.relu(self.stem_conv5(x)))
        x = self.drop_layer(x)

        x_left = self.block_a18_bn_left1(F.relu(self.block_a18_conv_left1(x)))
        x_left = F.relu(self.block_a18_conv_left2(x_left))
        x_mid = self.block_a18_bn_mid1(F.relu(self.block_a18_conv_mid1(x)))
        x_mid = self.block_a18_bn_mid2(F.relu(self.block_a18_conv_mid2(x_mid)))
        x_mid = F.relu(self.block_a18_conv_mid3(x_mid))
        x_right = F.relu(self.block_a18_conv_right1(x))
        x = self.block_a18_bn_cat(torch.cat([x_left, x_mid, x_right], 1))
        x = self.drop_layer(x)

        x_left = self.block_a24_bn_left1(F.relu(self.block_a24_conv_left1(x)))
        x_left = F.relu(self.block_a24_conv_left2(x_left))
        x_mid = self.block_a24_bn_mid1(F.relu(self.block_a24_conv_mid1(x)))
        x_mid = self.block_a24_bn_mid2(F.relu(self.block_a24_conv_mid2(x_mid)))
        x_mid = F.relu(self.block_a24_conv_mid3(x_mid))
        x_right = F.relu(self.block_a24_conv_right1(x))
        x = self.block_a24_bn_cat(torch.cat([x_left, x_mid, x_right], 1))
        x = self.drop_layer(x)

        """x_left = F.relu(self.block_b_conv_left1(x))        
        x_mid = self.block_b_bn_mid1(F.relu(self.block_b_conv_mid1(x)))
        x_mid = self.block_b_bn_mid2(F.relu(self.block_b_conv_mid2(x_mid)))
        x_mid = F.relu(self.block_b_conv_mid3(x_mid))
        x = self.block_b_bn_cat(torch.cat([x_left,x_mid],1))
        x = self.drop_layer(x)"""

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.softmax(self.out_layer(x), dim=1)

        return x