import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

class GumbelSigmoid(nn.Module):
    def __init__(self, tau=1.0):
        super(GumbelSigmoid, self).__init__()

        self.tau = tau
        self.softmax = nn.Softmax(dim=1)
        self.p_value = 1e-8

    def forward(self, x, is_eval=False):
        r = 1 - x

        x = (x + self.p_value).log()
        r = (r + self.p_value).log()

        if not is_eval:
            x_N = torch.rand_like(x)
            r_N = torch.rand_like(r)
        else:
            x_N = 0.5 * torch.ones_like(x)
            r_N = 0.5 * torch.ones_like(r)

        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()
        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()

        x = x + x_N
        x = x / (self.tau + self.p_value)
        r = r + r_N
        r = r / (self.tau + self.p_value)

        x = torch.cat((x, r), dim=1)
        x = self.softmax(x)

        return x

class Separation(torch.nn.Module):
    # 分离鲁棒特征与非鲁棒特征网络
    def __init__(self, size, num_channel=64, tau=0.1):
        super(Separation, self).__init__()
        C, H, W = size
        self.C, self.H, self.W = C, H, W
        self.tau = tau

        # 可学习的参数
        self.sep_net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, num_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, feat, is_eval=False):
        rob_map = self.sep_net(feat)

        mask = rob_map.reshape(rob_map.shape[0], 1, -1)
        mask = torch.nn.Sigmoid()(mask)
        mask = GumbelSigmoid(tau=self.tau)(mask, is_eval=is_eval)
        mask = mask[:, 0].reshape(mask.shape[0], self.C, self.H, self.W) # 分离鲁棒与非鲁棒特征的掩模

        r_feat = feat * mask # 鲁棒特征
        nr_feat = feat * (1 - mask) # 非鲁棒特征

        return r_feat, nr_feat, mask


class Recalibration(nn.Module):
    def __init__(self, size, num_channel=64):
        super(Recalibration, self).__init__()
        C, H, W = size
        self.rec_net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, num_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, nr_feat, mask):
        rec_units = self.rec_net(nr_feat) 
        rec_units = rec_units * (1 - mask) # 校正后的特征激活在乘上非鲁棒性的掩模
        rec_feat = nr_feat + rec_units

        return rec_feat


class ResNetFSR(nn.Module):
    def __init__(self, args, num_classes=10, image_size=(28, 28), tau=0.1):
        super(ResNetFSR, self).__init__()
        
        self.args = args
        self.image_size = image_size
        self.tau = tau

        # 加载 ResNet18 模型
        resnet18 = torchvision.models.resnet18(pretrained=False)
        if args.dataset == "Cifar10" or args.dataset == "Cifar100":
            resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

        resnet18.maxpool = nn.Identity()
        
        # 特征提取器
        self.backbone = nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3,
            resnet18.layer4,
            resnet18.avgpool
        )

        # 适配器
        self.separation = Separation(size=(512, 1, 1), tau=self.tau)
        self.recalibration = Recalibration(size=(512, 1, 1))
        self.aux = nn.Sequential(nn.Linear(512, num_classes))

        # 线性分类器
        self.linear = nn.Linear(resnet18.fc.in_features, num_classes)

    def forward(self, x, is_eval=False):

        r_outputs = []
        nr_outputs = []
        rec_outputs = []

        out = self.backbone(x)
        
        # 鲁棒性特征r_feat/nr_feat
        r_feat, nr_feat, mask = self.separation(out, is_eval=is_eval)
        r_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(r_feat).reshape(r_feat.shape[0], -1))
        r_outputs.append(r_out)
        nr_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(nr_feat).reshape(nr_feat.shape[0], -1))
        nr_outputs.append(nr_out)

        # nr_feat送入校正网络
        # rec_feat = self.recalibration(nr_feat, mask)
        # rec_out = self.aux(torch.nn.AdaptiveAvgPool2d(1)(rec_feat).reshape(rec_feat.shape[0], -1))
        # rec_outputs.append(rec_out)

        # 鲁棒特征+校正 + rec_feat 
        out = r_feat 

        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        #  
        return out, r_outputs, nr_outputs, rec_outputs
        

class ResNet(nn.Module):
    def __init__(self, args, num_classes=10, image_size=(28, 28)):
        super(ResNet, self).__init__()
        
        self.args = args
        self.image_size = image_size

        # 加载 ResNet18 模型
        resnet18 = torchvision.models.resnet18(pretrained=False)
        if args.dataset == "Cifar10" or args.dataset == "Cifar100":
            resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

        resnet18.maxpool = nn.Identity()
        # 特征提取器
        self.backbone = nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3,
            resnet18.layer4,
            resnet18.avgpool
        )

        # 线性分类器
        self.linear = nn.Linear(resnet18.fc.in_features, num_classes)

    def forward(self, x):

        out = self.backbone(x)
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out



def ResNet18FSR(args, num_classes=10, image_size=(28, 28), tau=0.1):
    return ResNetFSR(args, num_classes=num_classes, image_size=image_size, tau=tau)

def ResNet18(args, num_classes=10, image_size=(28, 28)):
    return ResNet(args, num_classes=num_classes, image_size=image_size)

if __name__ == "__main__":
    # model = ResNet18()
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    # x = torch.randn(10,3,32,32)
    # y = model(x)
    # print(y)    
    # for name, param in model.state_dict().items():
    #     print(name)
    print(list(model.children()))