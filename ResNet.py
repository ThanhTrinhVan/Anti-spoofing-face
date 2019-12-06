import torch
import torch.nn as nn
import time
from torchvision import models

class BasicModule(nn.Module):
    def __init__(self, opt=None):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def save(self, name=None):
        if name is None:
            prefix = './checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

class MyresNet18(BasicModule):
    def __init__(self):
        super(MyresNet18, self).__init__()
        model = models.resnet18(pretrained=True)
        self.resnet_lay = nn.Sequential(*list(model.children())[:-2])
        self.conv1_lay = nn.Conv2d(512, 256, kernel_size=(1,1), stride=(1,1))
        self.relu1_lay = nn.ReLU(inplace=True)
        self.drop_lay = nn.Dropout2d(0.5)
        self.global_average = nn.AdaptiveAvgPool2d((1,1))
        self.fc_Linear_lay2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.resnet_lay(x)
        x = self.conv1_lay(x)
        x = self.relu1_lay(x)
        x = self.drop_lay(x)
        x = self.global_average(x)
        x = x.view(x.size(0), -1)
        x = self.fc_Linear_lay2(x)
        return x