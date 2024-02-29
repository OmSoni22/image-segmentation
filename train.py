import os
import sys
import argparse
from typing import List, AnyStr
import json

import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from torch import Tensor
import torch.nn as nn
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split


from unet import UNet
from utils.dice_ import dice_loss
from utils.dataloader import Dataset_segmentation


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", default=1e-5, type=float, help="learning rate")
    ap.add_argument("--in_ch", default=3, type=int, help="Output channel")
    ap.add_argument("--out_ch", default=2, type=int, help="input channel")
    ap.add_argument("--img_dir", required=True, help="img directory")
    ap.add_argument("--mask_dir", required=True, help="mask directory")
    ap.add_argument("--img_size", default=512, type=int, help="img directory")
    ap.add_argument("--batch_size", default=32, type=int, help="batch size for training")
    ap.add_argument("--num_worker", default=4, type=int, help="number of worker")
    ap.add_argument("--epochs", default=25, type=int, help="number of epochs")
    ap.add_argument("-l", "--log_dir", default="logs", help="Logs dir for tensorboard")
    ap.add_argument("-t", "--train_dir", default="train_dir", help="Checkpoint directory")
    ap.add_argument("-c", "--checkpoint", default=None, help="Checkpoint for agent")
    ap.add_argument("-g", "--gpus", default=None, help="Provide GPU IDs for training")
    ap.add_argument("--logs", default="logs", help="Logs directory")  # Add this line
    ap.add_argument("--alpha", default=0.5, type=float, help="weightage of bce loss")
    ap.add_argument("--beta", default=0.4, type=float, help="beta to penelize FN or FP")
    return ap.parse_args()


def get_gpu_ids(
        gpu_id_string: AnyStr,
) -> List:
    return [int(i) for i in gpu_id_string.split(",") if len(i) > 0]


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def bce_(pred: Tensor, target: Tensor, beta: float , smooth: float = 1e-6):

    bce_loss = -torch.mean(beta * target * torch.log(pred + 1e-6) + (1 - beta) * (1 - target) * torch.log(1 - pred + smooth))

    return bce_loss



def train(opt, net, device) -> None:
    """
    Training function
    :param opt: argparse object for arguments
    :param net: model to train
    :param device: training device
    :param criterion: loss criteria
    :return: None
    """

    transforms_img = transforms.Compose([transforms.ColorJitter(brightness=0.8),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])

    transforms_mask = transforms.Compose([transforms.ToTensor(), ])

    dataset = Dataset_segmentation(opt.img_dir,
                                       opt.mask_dir,
                                       random_crop=False,
                                       image_size=opt.img_size)

    train_loader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.num_worker,
                            pin_memory=True)
    
    optimizer = optim.Adam(net.parameters(),
                           lr=opt.lr,
                           betas=(0.99, 0.999),
                           weight_decay=0.0005)

    scheduler = StepLR(optimizer,
                       step_size=10,
                       gamma=0.9)

    epochs = opt.epochs
    net.train()

    # criterion = nn.BCEWithLogitsLoss()
    min_loss = 10.0

    alpha = opt.alpha
    beta = opt.beta
    

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        train_loss = 0
        train_step = 0

        for data in tqdm(train_loader):
            train_step+=1
            images, true_masks = data

            images = images.to(device)
            true_masks = true_masks.to(device)
            masks_pred = net(images)

            bce = bce_(masks_pred.squeeze(1), true_masks.float(), beta)
            dice_ = dice_loss(masks_pred.squeeze(1), true_masks.float(), multiclass=False)
            loss = alpha*bce + (1 - alpha)*dice_
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        scheduler.step()
        train_loss_avg = train_loss / train_step

        print('Training Loss: {}'.format(train_loss_avg))

        if (epoch+1)%30==0 or epoch==0 or epoch==epochs:
            torch.save(net.state_dict(),
                   os.path.join(opt.train_dir, 'CP_{}.pth'.format(epoch + 1)))

            print('Checkpoint {} saved !'.format(epoch + 1))
        

        if train_loss_avg < min_loss:
            min_loss = train_loss_avg
            torch.save(net.state_dict(), os.path.join(opt.train_dir, 'best_CP.pth'))
            print('Best Epochs Weight Saved')
        

def main():
    opt = get_args()
    
    # CAI: set primary GPU device
    device = torch.device('cuda')
    
    # CAI: preparing directories
    os.makedirs(opt.train_dir, exist_ok=True)
    os.makedirs(opt.logs, exist_ok=True)

    # CAI: prepare model
    net = UNet(opt.in_ch, opt.out_ch).to(device)

    try:
        train(opt=opt,
              net=net,
              device=device,
            )

    except KeyboardInterrupt:
        torch.save(net.state_dict(), os.path.join(opt.train_dir,'INTERRUPTED.pth'))
        print('Saved interrupt')
        sys.exit(0)


if __name__ == '__main__':
    main()

# python3 train.py --lr 0.001 --in_ch 3 --out_ch 7 --img_dir Pytorch-UNet/data/imgs --mask_dir Pytorch-UNet/data/new_masks --batch_size 2 --epochs 300 -t /TrainwtBeta02 --beta 0.2