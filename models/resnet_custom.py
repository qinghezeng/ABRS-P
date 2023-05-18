# modified from Pytorch official resnet.py
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torchsummary import summary
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Bottleneck_Baseline(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_Baseline, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_Baseline(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet_Baseline, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

def resnet50_baseline(pretrained=False):
    """Constructs a Modified ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # use adaptive mean-spatial pooling after the 3rd residual block of the network, so 3 stages (/blocks) with 3,4,6 Bottleneck_Baseline, respectively
    model = ResNet_Baseline(Bottleneck_Baseline, [3, 4, 6, 3]) 
    if pretrained:
        model = load_pretrained_weights(model, 'resnet50')
    return model

def load_pretrained_weights(model, name):
    pretrained_dict = model_zoo.load_url(model_urls[name])
    model.load_state_dict(pretrained_dict, strict=False)
    return model

#******************************************************************************
# Add by Qinghe 07/11/2020

from collections import OrderedDict
import torchvision

def resnet18_torchvision(pretrained=False):
    """ resnet18 from torchvision
    """
    model = torchvision.models.resnet18(pretrained=pretrained)
    
    container_names = []
    for name, module in model.named_children():
        container_names.append(name)
    # newmodel = torch.nn.Sequential(*list(model_conv.children())[:-1])
    model = torch.nn.Sequential(OrderedDict(zip(container_names[:-1], list(model.children())[:-1])))
    
    return model

def resnet34_torchvision(pretrained=False):
    """ resnet34 from torchvision
        one more block compared to the clam-modified one
    """
    model = torchvision.models.resnet34(pretrained=pretrained)
    
    container_names = []
    for name, module in model.named_children():
        container_names.append(name)
    # newmodel = torch.nn.Sequential(*list(model_conv.children())[:-1])
    model = torch.nn.Sequential(OrderedDict(zip(container_names[:-1], list(model.children())[:-1])))
    
    return model

def resnet152_torchvision(pretrained=False):
    """ resnet152 from torchvision
    """
    model = torchvision.models.resnet152(pretrained=pretrained)
    
    container_names = []
    for name, module in model.named_children():
        container_names.append(name)
    del(container_names[-1])
    del(container_names[-2])
    model_children = list(model.children())
    del(model_children[-3:])
    resnet50 = resnet50_baseline(pretrained=pretrained)
    model_children.append(list(resnet50.children())[-1])

    # newmodel = torch.nn.Sequential(*list(model_conv.children())[:-1])
    model = torch.nn.Sequential(OrderedDict(zip(container_names, model_children)))
    
    return model

def densenet121_torchvision(pretrained=False):
    """ desnet121 from torchvision
        Have to first modify "/home/visiopharm5/anaconda3/envs/clam/lib/python3.7/site-packages/torchvision/models/densenet.py"
        Otherwise will get a KeyError: 'module name can\'t contain "."'
    """
    model = torchvision.models.densenet121(pretrained=pretrained)
    
    container_names = []
    for name, module in model.named_children():
        container_names.append(name) # features, classifier

    # newmodel = torch.nn.Sequential(*list(model_conv.children())[:-1])
    model = torch.nn.Sequential(OrderedDict(zip(container_names[:-1], list(model.children())[:-1]))) # remove the classifier layer
    
    return model

# print(resnet152_torchvision(False))
    
##### 26/11/2021
    
def resnet18_simclr():
    MODEL_PATH = 'models/simclr/tenpercent_resnet18.pt'
    RETURN_PREACTIVATION = True  # return features from the model, if false return classification logits
    NUM_CLASSES = 2  # only used if RETURN_PREACTIVATION = False
    
    
    def load_model_weights(model, weights):
    
        model_dict = model.state_dict()
        weights = {k: v for k, v in weights.items() if k in model_dict}
        if weights == {}:
            print('No weight could be loaded..')
        model_dict.update(weights)
        model.load_state_dict(model_dict)
    
        return model
    
    
    model = torchvision.models.__dict__['resnet18'](pretrained=False)
    
    state = torch.load(MODEL_PATH, map_location='cuda:0')
    
    state_dict = state['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
    
    model = load_model_weights(model, state_dict)
    
    if RETURN_PREACTIVATION:
        model.fc = torch.nn.Sequential()
    else:
        model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    
    model = model.cuda()
        
