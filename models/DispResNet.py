import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, output_channels // 4, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels // 4, output_channels // 4, 3, stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels // 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_channels // 4, output_channels, 1, 1, bias=False)
        self.conv4 = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride != 1):
            residual = self.conv4(out1)
        out += residual
        return out


class AttentionModule_stage1(nn.Module):
    # input size is 32 * 104
    def __init__(self, in_channels, out_channels, size1=(32, 104), size2=(16, 52), size3=(8, 26)):
        super(AttentionModule_stage1, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)
        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = ResidualBlock(in_channels, out_channels)
        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax2_blocks = ResidualBlock(in_channels, out_channels)
        self.skip2_connection_residual_block = ResidualBlock(in_channels, out_channels)
        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax3_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )
        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)
        self.softmax4_blocks = ResidualBlock(in_channels, out_channels)
        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
        self.softmax5_blocks = ResidualBlock(in_channels, out_channels)
        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)
        self.softmax6_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels , kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels , kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )
        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
        out_mpool3 = self.mpool3(out_softmax2)
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        out_interp3 = self.interpolation3(out_softmax3) + out_softmax2
        out = out_interp3 + out_skip2_connection
        out_softmax4 = self.softmax4_blocks(out)
        out_interp2 = self.interpolation2(out_softmax4) + out_softmax1
        out = out_interp2 + out_skip1_connection
        out_softmax5 = self.softmax5_blocks(out)
        out_interp1 = self.interpolation1(out_softmax5) + out_trunk
        out_softmax6 = self.softmax6_blocks(out_interp1)
        out = (1 + out_softmax6) * out_trunk
        out_last = self.last_blocks(out)

        return out_last


class AttentionModule_stage2(nn.Module):
    # input image size is 16*52
    def __init__(self, in_channels, out_channels, size1=(16, 52), size2=(8, 26)):
        super(AttentionModule_stage2, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax1_blocks = ResidualBlock(in_channels, out_channels)

        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.softmax2_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )

        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)

        self.softmax3_blocks = ResidualBlock(in_channels, out_channels)

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax4_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)

        out_interp2 = self.interpolation2(out_softmax2) + out_softmax1
        # print(out_skip2_connection.data)
        # print(out_interp3.data)
        out = out_interp2 + out_skip1_connection
        out_softmax3 = self.softmax3_blocks(out)
        out_interp1 = self.interpolation1(out_softmax3) + out_trunk
        out_softmax4 = self.softmax4_blocks(out_interp1)
        out = (1 + out_softmax4) * out_trunk
        out_last = self.last_blocks(out)

        return out_last


class AttentionModule_stage3(nn.Module):
    # input image size is 8*26
    def __init__(self, in_channels, out_channels, size1=(8, 26)):
        super(AttentionModule_stage3, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
         )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels)
        )

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax2_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)

        out_interp1 = self.interpolation1(out_softmax1) + out_trunk
        out_softmax2 = self.softmax2_blocks(out_interp1)
        out = (1 + out_softmax2) * out_trunk
        out_last = self.last_blocks(out)

        return out_last


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def make_layer(inplanes, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)


def downsample_conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
    )


def predict_disp(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )


def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )


def crop_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]


class DispResNet(nn.Module):

    def __init__(self, alpha=10, beta=0.01):
        super(DispResNet, self).__init__()

        self.alpha = alpha
        self.beta = beta

        att_planes = [128, 256, 512]
        self.attention_module1 = AttentionModule_stage1(att_planes[0], att_planes[0])
        self.attention_module2_1 = AttentionModule_stage2(att_planes[1], att_planes[1])
        self.attention_module2_2 = AttentionModule_stage2(att_planes[1], att_planes[1])
        self.attention_module3_1 = AttentionModule_stage3(att_planes[2], att_planes[2])
        self.attention_module3_2 = AttentionModule_stage3(att_planes[2], att_planes[2])
        self.attention_module3_3 = AttentionModule_stage3(att_planes[2], att_planes[2])

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        self.conv1 = downsample_conv(3,              conv_planes[0], kernel_size=7)
        self.conv2 = make_layer(conv_planes[0], BasicBlock, conv_planes[1], blocks=2, stride=2)
        self.conv3 = make_layer(conv_planes[1], BasicBlock, conv_planes[2], blocks=2, stride=2)
        self.conv4 = make_layer(conv_planes[2], BasicBlock, conv_planes[3], blocks=3, stride=2)
        self.conv5 = make_layer(conv_planes[3], BasicBlock, conv_planes[4], blocks=3, stride=2)
        self.conv6 = make_layer(conv_planes[4], BasicBlock, conv_planes[5], blocks=3, stride=2)
        self.conv7 = make_layer(conv_planes[5], BasicBlock, conv_planes[6], blocks=3, stride=2)

        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = upconv(conv_planes[6],   upconv_planes[0])
        self.upconv6 = upconv(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconv(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconv(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconv(upconv_planes[5], upconv_planes[6])

        self.iconv7 = make_layer(upconv_planes[0] + conv_planes[5], BasicBlock, upconv_planes[0], blocks=2, stride=1)
        self.iconv6 = make_layer(upconv_planes[1] + conv_planes[4], BasicBlock, upconv_planes[1], blocks=2, stride=1)
        self.iconv5 = make_layer(upconv_planes[2] + conv_planes[3], BasicBlock, upconv_planes[2], blocks=2, stride=1)
        self.iconv4 = make_layer(upconv_planes[3] + conv_planes[2], BasicBlock, upconv_planes[3], blocks=2, stride=1)
        self.iconv3 = make_layer(1 + upconv_planes[4] + conv_planes[1], BasicBlock, upconv_planes[4], blocks=1, stride=1)
        self.iconv2 = make_layer(1 + upconv_planes[5] + conv_planes[0], BasicBlock, upconv_planes[5], blocks=1, stride=1)
        self.iconv1 = make_layer(1 + upconv_planes[6], BasicBlock, upconv_planes[6], blocks=1, stride=1)

        self.predict_disp6 = predict_disp(upconv_planes[1])
        self.predict_disp5 = predict_disp(upconv_planes[2])
        self.predict_disp4 = predict_disp(upconv_planes[3])
        self.predict_disp3 = predict_disp(upconv_planes[4])
        self.predict_disp2 = predict_disp(upconv_planes[5])
        self.predict_disp1 = predict_disp(upconv_planes[6])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out_conv1 = self.conv1(x)  # input(128, 416), output(64, 208)
        out_conv2 = self.conv2(out_conv1)  # input(64, 128), output(32, 52)
        out_conv3 = self.conv3(out_conv2)
        out_attention3 = self.attention_module1(out_conv3)
        out_conv4 = self.conv4(out_attention3)
        out_attention4_1 = self.attention_module2_1(out_conv4)
        out_attention4_2 = self.attention_module2_2(out_attention4_1)
        out_conv5 = self.conv5(out_attention4_2)
        out_attention5_1 = self.attention_module3_1(out_conv5)
        out_attention5_2 = self.attention_module3_2(out_attention5_1)
        out_attention5_3 = self.attention_module3_3(out_attention5_2)
        out_conv6 = self.conv6(out_attention5_3)
        out_conv7 = self.conv7(out_conv6)

        out_upconv7 = crop_like(self.upconv7(out_conv7), out_conv6)
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)

        out_upconv6 = crop_like(self.upconv6(out_iconv7), out_attention5_3)
        concat6 = torch.cat((out_upconv6, out_attention5_3), 1)
        out_iconv6 = self.iconv6(concat6)
        disp6 = self.alpha * self.predict_disp6(out_iconv6) + self.beta

        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_attention4_2)
        concat5 = torch.cat((out_upconv5, out_attention4_2), 1)
        out_iconv5 = self.iconv5(concat5)
        disp5 = self.alpha * self.predict_disp5(out_iconv5) + self.beta

        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_attention3)
        concat4 = torch.cat((out_upconv4, out_attention3), 1)
        out_iconv4 = self.iconv4(concat4)
        disp4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta

        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up = crop_like(F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        disp3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta

        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = crop_like(F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        disp2 = self.alpha * self.predict_disp2(out_iconv2) + self.beta

        out_upconv1 = crop_like(self.upconv1(out_iconv2), x)
        disp2_up = crop_like(F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        out_iconv1 = self.iconv1(concat1)
        disp1 = self.alpha * self.predict_disp1(out_iconv1) + self.beta

        if self.training:
            return disp1, disp2, disp3, disp4, disp5, disp6
        else:
            return disp1
