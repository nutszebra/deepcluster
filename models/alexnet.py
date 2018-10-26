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

    def __init__(self, features, num_classes, sobel, length_train, alpha, memory_dim, momentum, reassign_period, beta):
        super(AlexNet, self).__init__()
        print('num_classes: {}'.format(num_classes))
        print('sobel: {}'.format(sobel))
        print('alpha: {}'.format(alpha))
        self.alpha, self.momentum, self.reassign_period, self.beta = alpha, momentum, reassign_period, beta
        # define embedding
        self.embedding = nn.Embedding(memory_dim, num_classes)
        # matrix to record whether embedding is used or not while training
        self.register_buffer('history', torch.randn(memory_dim))
        self.reset_history()
        self.counter = 0
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
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def reset_history(self):
        self.history.zero_()
        self.counter = 0

    def update_memory(self, x1, x2, momentum):
        return (momentum * x1 + (1.0 - momentum) * x2).detach()

    def reassign(self):
        if ((self.history == 0).sum().item() == 0) or (self.history.sum().item() == 0):
            print('No reassignment')
            # every memory are used at least once or no history, so no assignment
            pass
        else:
            # index of used embedding and non-used ones
            used_embedding = torch.nonzero(self.history).squeeze(1)
            unused_embedding = torch.nonzero(self.history == 0).squeeze(1)
            print('Reassignment: {}'.format(len(unused_embedding.data.cpu().numpy())))
            for i in unused_embedding:
                selected_embedding = self.embedding.weight[used_embedding[random.randint(0, len(used_embedding) - 1)]]
                self.embedding.weight[i].data = self.update_memory(selected_embedding, torch.randn_like(selected_embedding), self.momentum)
        self.reset_history()

    def crit(self, y, t):
        # reassing embedding
        if self.training is True:
            self.counter += 1
            if self.counter >= self.reassign_period:
                self.reassign()
        # index of embedding that are nearest
        index_of_min_embedding = torch.argmax(torch.matmul(F.softmax(y, 1), F.softmax(self.embedding.weight, 1).transpose(0, 1)), 1)
        print(index_of_min_embedding)
        if self.training is True:
            # update history while training
            self.history[index_of_min_embedding] += 1
        t = self.embedding.weight[index_of_min_embedding]
        loss = cross_entropy.softmax_cross_entropy(y, F.softmax(t, 1), average=True, reduce=True)
        loss_push = cross_entropy.softmax_cross_entropy(torch.cat((t[1:], t[:1])), F.softmax(t, 1), average=True, reduce=True)
        loss_push2 = cross_entropy.softmax_cross_entropy(torch.cat((y[1:], y[:1])), F.softmax(y, 1), average=True, reduce=True)
        return loss - self.alpha * loss_push - self.beta * loss_push2


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


def alexnet(sobel=True, bn=True, out=32, length_train=None, alpha=1.0e-2, memory_dim=10000, momentum=0.99, reassign_period=200, beta=10.0):
    dim = 2 + int(not sobel)
    model = AlexNet(make_layers_features(CFG['2012'], dim, bn=bn), out, sobel, length_train, alpha, memory_dim, momentum, reassign_period, beta)
    return model
