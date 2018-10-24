import argparse
import os
import subprocess
import tqdm

import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets

import modify_dataset
import models

from util import AverageMeter

parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                    choices=['alexnet', 'vgg16'], default='alexnet',
                    help='CNN architecture (default: alexnet)')
parser.add_argument('--sobel', default=True, action='store_true', help='Sobel filtering')
parser.add_argument('--lr', default=1.0e-4, type=float,
                    help='learning rate (default: 0.05)')
parser.add_argument('--weight_decay', default=1.0e-5, type=float,
                    help='weight decay pow (default: -5)')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts) (default: 0)')
parser.add_argument('--batch', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: None)')
parser.add_argument('--checkpoints', type=int, default=25000,
                    help='how many iterations between two checkpoints (default: 25000)')
parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
parser.add_argument('--exp', type=str, default='', help='path to exp folder')
parser.add_argument('--verbose', action='store_true', help='chatty')


def main():
    global args
    args = parser.parse_args()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # load imgs
    dataset = datasets.ImageFolder(args.data)
    image_lists, _ = zip(*dataset.imgs)

    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch))
    model = models.__dict__[args.arch](sobel=args.sobel, length_train=len(image_lists))
    model.cuda()
    cudnn.benchmark = True

    # create optimizer
    # optimizer = torch.optim.SGD(
    #     filter(lambda x: x.requires_grad, model.parameters()),
    #     lr=args.lr,
    #     momentum=args.momentum,
    #     weight_decay=10**args.wd,
    # )
    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # creating checkpoint repo
    exp_check = os.path.join(args.exp, 'checkpoints')
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    # training convnet with DeepCluster
    hash_git = subprocess.check_output('git log -n 1', shell=True).decode('utf-8').split(' ')[1].split('\n')[0]
    for epoch in tqdm.tqdm(range(args.start_epoch, args.epochs), desc=hash_git, leave=False, ncols=80):
        # create dataset
        train_dataset = modify_dataset.create_dataset(image_lists)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch,
            num_workers=args.workers,
            shuffle=True,
            pin_memory=True,
        )

        loss = train(train_dataloader, model, optimizer, epoch)

        # print log
        if args.verbose:
            print('###### Epoch [{0}] ###### \n'
                  'ConvNet loss: {1:.3f}'
                  .format(epoch, loss))
            print('####################### \n')
        # save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   os.path.join(args.exp, 'checkpoint.pth.tar'))


def train(loader, model, opt, epoch):
    # logger
    losses = AverageMeter()
    # switch to train mode
    model.train()
    for i, (input_tensor, target) in enumerate(tqdm.tqdm(loader, desc='train {} epochs'.format(epoch), leave=True, ncols=80)):
        # save checkpoint
        n = len(loader) * epoch + i
        if n % args.checkpoints == 0:
            path = os.path.join(
                args.exp,
                'checkpoints',
                'checkpoint_' + str(n / args.checkpoints) + '.pth.tar',
            )
            if args.verbose:
                print('Save checkpoint at: {0}'.format(path))
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': opt.state_dict()
            }, path)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input_tensor.cuda())
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = model.crit(output, target_var)

        # record loss
        losses.update(loss.data[0], input_tensor.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        loss.backward()
        opt.step()

        if args.verbose and (i % 200) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), loss=losses))
    return losses.avg


if __name__ == '__main__':
    main()
