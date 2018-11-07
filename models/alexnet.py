import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import cross_entropy

__all__ = ['AlexNet', 'alexnet']

# (number of filters, kernel size, stride, pad)
CFG = {
    '2012': [(96, 11, 4, 2), 'M', (256, 5, 1, 2), 'M', (384, 3, 1, 1), (384, 3, 1, 1), (256, 3, 1, 1), 'M']
}


class AlexNet(nn.Module):

    def __init__(self, features, num_classes, sobel, alpha, multi):
        super(AlexNet, self).__init__()
        print('num_classes: {}'.format(num_classes))
        print('sobel: {}'.format(sobel))
        print('alpha: {}'.format(alpha))
        print('multi: {}'.format(multi))
        self.alpha, self.multi = alpha, multi
        # define classifier
        self.features = features
        self.classifier = nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(256 * 6 * 6, 4096),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(inplace=True))
        self.top_layer = nn.Linear(4096, num_classes)
        self._initialize_weights()

        if sobel:
            grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
            grayscale.weight.data.fill_(1.0 / 3.0)
            grayscale.bias.data.zero_()
            sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
            sobel_filter.weight.data[0, 0].copy_(
                torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            )
            sobel_filter.weight.data[1, 0].copy_(
                torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            )
            sobel_filter.bias.data.zero_()
            self.sobel = nn.Sequential(grayscale, sobel_filter)
            for p in self.sobel.parameters():
                p.requires_grad = False
        else:
            self.sobel = None

    def forward(self, x):
        if self.sobel:
            x = self.sobel(x)
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = self.top_layer(x)
        return x

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

    def crit(self, y):
        batch = int(y.shape[0] / 2)
        y1, y2 = y[:batch], y[batch:]
        y1, y2 = y1.view(batch * self.multi, -1), y2.view(batch * self.multi, -1)
        loss_pull = cross_entropy.softmax_cross_entropy(y1, F.softmax(y2, 1).detach(), average=True, reduce=True)
        y = y.view(y.shape[0] * self.multi, -1)
        loss_push = torch.clamp(cross_entropy.softmax_cross_entropy(torch.cat((y[1:], y[:1])), F.softmax(y, 1).detach(), average=True, reduce=False), 0.0, 2.0).sum()
        return loss_pull - self.alpha * loss_push


def make_layers_features(cfg, input_dim, bn):
    layers = []
    in_channels = input_dim
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v[0], kernel_size=v[1], stride=v[2], padding=v[3])
            if bn:
                layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v[0]
    return nn.Sequential(*layers)


def alexnet(sobel=False, bn=True, out=100, alpha=1.0e-1, multi=10):
    dim = 2 + int(not sobel)
    model = AlexNet(make_layers_features(CFG['2012'], dim, bn=bn), out, sobel, alpha, multi)
    return model
