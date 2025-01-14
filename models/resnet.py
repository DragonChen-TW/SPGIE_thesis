import torch
from torch import nn
from torch.nn import functional as f

def conv1x1(in_chn, out_chn, stride=1):
    return nn.Conv2d(in_chn, out_chn, kernel_size=1,
                     stride=stride, bias=False)

def conv3x3(in_chn, out_chn, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_chn, out_chn, kernel_size=3,
                     stride=stride, padding=dilation,
                     groups=groups, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_chn, out_chn, stride=1, downsample=None, groups=1, base_width=64):
        super().__init__()
        self.layers = nn.Sequential(
            conv3x3(in_chn, out_chn, stride),
            nn.BatchNorm2d(out_chn),
            nn.ReLU(inplace=True),
            conv3x3(out_chn, out_chn),
            nn.BatchNorm2d(out_chn),
        )
        self.downsample = downsample

    def forward(self, x):
        out = self.layers(x)
        
        identity = x
        # use 1-channel conv to reduce channels
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = f.relu(out)

        return out

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_chn, out_chn, stride=1, downsample=None, groups=1, base_width=64):
        super().__init__()

        width = int(out_chn * (base_width / 64.)) * groups
        self.layers = nn.Sequential(
            conv1x1(in_chn, width),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            
            conv3x3(width, width, stride=stride, groups=groups),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            
            conv1x1(width, out_chn * self.expansion),
            nn.BatchNorm2d(out_chn * self.expansion),
        )
        self.downsample = downsample

    def forward(self, x):
        out = self.layers(x)
        
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = f.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, pre, num_classes=10):
        super().__init__()
        
        self.in_chn = 16
        self.base_width = 16
        
        self.pre = pre
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, out_chn, block_num, stride=1):
        downsample = None
        
        if stride != 1 or self.in_chn != out_chn * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_chn, out_chn * block.expansion, stride),
                nn.BatchNorm2d(out_chn * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_chn, out_chn, stride,
                            downsample,
                            base_width=self.base_width))
        self.in_chn = out_chn * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_chn, out_chn,
                                base_width=self.base_width))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.max_pool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class ResNetOriginal(nn.Module):
    def __init__(self, block, layers, pre, num_classes=10):
        super().__init__()
        
        self.in_chn = 64
        self.base_width = 64
        
        self.pre = pre
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_chn, block_num, stride=1):
        downsample = None
        
        if stride != 1 or self.in_chn != out_chn * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_chn, out_chn * block.expansion, stride),
                nn.BatchNorm2d(out_chn * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_chn, out_chn, stride,
                            downsample,
                            1, self.base_width)) # 1x64
        self.in_chn = out_chn * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_chn, out_chn,
                                1, base_width=self.base_width))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class ResNext(nn.Module):
    def __init__(self, block, layers, pre, num_classes=10, groups=1, width_per_group=64):
        super().__init__()
        
        self.in_chn = 64
        self.groups = groups
        self.base_width = width_per_group
        
        self.pre = pre
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_chn, block_num, stride=1):
        downsample = None
        
        if stride != 1 or self.in_chn != out_chn * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_chn, out_chn * block.expansion, stride),
                nn.BatchNorm2d(out_chn * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_chn, out_chn, stride,
                            downsample, self.groups,
                            self.base_width))
        self.in_chn = out_chn * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_chn, out_chn, 
                                groups=self.groups,
                                base_width=self.base_width))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        
        # print('pre', x.shape)
        
        x = self.layer1(x)
        # print('l1', x.shape)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        
        x = self.avg_pool(x)
        # print('l4+pool', x.shape)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

cifar10_pre = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=2,
              padding=3, bias=False),
    nn.BatchNorm2d(16),
    nn.ReLU(inplace=True),
)

def resnet20(pre=cifar10_pre):
    return ResNet(BasicBlock, [3, 3, 3], pre=pre)

def resnet32(pre=cifar10_pre):
    return ResNet(BasicBlock, [5, 5, 5], pre=pre)

def resnet44(pre=cifar10_pre):
    return ResNet(BasicBlock, [7, 7, 7], pre=pre)

def choose_original_pre(num_features=3):
    original_pre = nn.Sequential(
        nn.Conv2d(num_features, 64, kernel_size=3, stride=1,
                  padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
    )
    return original_pre

def resnet18(num_features=3):
    pre = choose_original_pre(num_features)
    return ResNetOriginal(BasicBlock, [2, 2, 2, 2], pre=pre)

def resnet34(num_features=3):
    pre = choose_original_pre(num_features)
    return ResNetOriginal(BasicBlock, [3, 4, 6, 3], pre=pre)

def resnet50(num_features=3):
    pre = choose_original_pre(num_features)
    return ResNetOriginal(BottleNeck, [3, 4, 6, 3], pre=pre)

def resnet77(num_features=3):
    pre = choose_original_pre(num_features)
    return ResNetOriginal(BottleNeck, [3, 4, 15, 3], pre=pre)

def resnext50_32x4d(num_features=3):
    pre = choose_original_pre(num_features)
    return ResNext(BottleNeck, [3, 4, 6, 3], pre=pre,
                  groups=32, width_per_group=4)

if __name__ == '__main__':
    model = resnext50_32x4d()
    x = torch.randn((16, 3, 32, 32))
    
    print(model(x).shape)
