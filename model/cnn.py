from torch import nn
import torch.nn.init as init
import torch.nn.functional as F

class NNet(nn.Module):
    def __init__(self,channel):
        super(SELayer,self).__init__()

        self.conv1  = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.fc1 = 
        self.fc2 = 
        self.dropout = 

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(F.max_pool2d(x, 2))

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(F.max_pool2d(x, 2))

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(F.max_pool2d(x, 2))

        x = x.view()
        x = self.fc1(x)
        x = self.fc2(x)

        return x 

def loss_fn_kd(outputs, labels, teacher_outputs):

    T = 1.0
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim = 1),)
