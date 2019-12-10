import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
from torch.nn.init import xavier_uniform_, zeros_


def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
        nn.BatchNorm2d(out_planes, eps=0.001),
        nn.ReLU(inplace=True)
    )


# 图像的size不改变
def CONV1(in_planes, out_planes, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


# 图像的size改变
def CONV2(in_planes, out_planes, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


class Inception_v1(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3_red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception_v1, self).__init__()
        # 1x1 conv branch
        self.branch1 = CONV1(in_planes, n1x1, kernel_size=1)
        # 1x1 conv -> 3x3 conv branch
        self.branch2 = nn.Sequential(
            CONV1(in_planes, n3x3_red, kernel_size=1),
            CONV1(n3x3_red, n3x3, kernel_size=3),
        )
        # 1x1 conv -> 3x3 conv ->3x3 branch
        self.branch3 = nn.Sequential(
            CONV1(in_planes, n5x5red, kernel_size=1),
            CONV1(n5x5red, n5x5, kernel_size=3),
            CONV1(n5x5, n5x5, kernel_size=3),
        )
        # 3x3_pool -> 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            CONV1(in_planes, pool_planes, kernel_size=1),
        )

    def forward(self, x):
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)
        return torch.cat([y1, y2, y3, y4], 1)


# inception v2, 图像的size改变
class Inception_v2(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3_red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception_v2, self).__init__()
        # 1x1 conv branch
        self.branch1 = CONV2(in_planes, n1x1, kernel_size=1)
        # 1x1 conv -> 3x3 conv branch
        self.branch2 = nn.Sequential(
            CONV1(in_planes, n3x3_red, kernel_size=1),
            CONV2(n3x3_red, n3x3, kernel_size=3),
        )
        # 1x1 conv -> 3x3 conv ->3x3 branch
        self.branch3 = nn.Sequential(
            CONV1(in_planes, n5x5red, kernel_size=1),
            CONV1(n5x5red, n5x5, kernel_size=3),
            CONV2(n5x5, n5x5, kernel_size=3),
        )
        # 3x3_pool -> 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            CONV1(in_planes, pool_planes, kernel_size=1),
        )

    def forward(self, x):
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
        conv_planes = [32, 64, 128, 128, 192, 192, 256, 256, 256, 256]
        self.conv1 = CONV2(6, conv_planes[0], kernel_size=7)
        self.conv2 = CONV2(conv_planes[0], conv_planes[1], kernel_size=5)
        self.Icp3 = Inception_v2(conv_planes[1], 40, 42, 56, 10, 16, 16)
        self.Icp4 = Inception_v1(conv_planes[2], 32, 48, 64, 10, 16, 16)
        self.Icp5 = Inception_v1(conv_planes[3], 64, 60, 80, 15, 24, 24)
        self.Icp6 = Inception_v2(conv_planes[4], 48, 72, 96, 15, 24, 24)
        self.Icp7 = Inception_v1(conv_planes[5], 96, 72, 96, 20, 32, 32)
        self.Icp8 = Inception_v1(conv_planes[6], 88, 78, 104, 20, 32, 32)
        self.Icp9 = Inception_v2(conv_planes[7], 80, 84, 112, 20, 32, 32)
        self.Icp10 = Inception_v2(conv_planes[8], 64, 96, 128, 20, 32, 32)
        self.conv11 = CONV2(conv_planes[9], 6, kernel_size=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, target_image, ref_img):
        input = [target_image, ref_img]
        input = torch.cat(input, 1)
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.Icp3(out_conv2)
        out_conv4 = self.Icp4(out_conv3)
        out_conv5 = self.Icp5(out_conv4)
        out_conv6 = self.Icp6(out_conv5)
        out_conv7 = self.Icp7(out_conv6)
        out_conv8 = self.Icp8(out_conv7)
        out_conv9 = self.Icp9(out_conv8)
        out_conv10 = self.Icp10(out_conv9)

        pose = self.conv11(out_conv10)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), 6)

        return pose
