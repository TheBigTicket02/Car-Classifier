import os
from argparse import ArgumentParser, Namespace
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.metrics.functional.classification import accuracy
from pytorch_lightning.loggers import WandbLogger
from torch import optim
#from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

import wandb

from collections import OrderedDict
from typing import Optional, Generator, Union
from torch.optim.optimizer import Optimizer
from torch.nn import Module

BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

#  --- Utility functions ---


def _make_trainable(module: Module) -> None:
    """Unfreezes a given module.
    Args:
        module: The module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module: Module,
                      train_bn: bool = True) -> None:
    """Freezes the layers of a given module.
    Args:
        module: The module to freeze
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze(module=child, train_bn=train_bn)


def freeze(module: Module,
           n: Optional[int] = None,
           train_bn: bool = True) -> None:
    """Freezes the layers up to index n (if n is not None).
    Args:
        module: The module to freeze (at least partially)
        n: Max depth at which we stop freezing the layers. If None, all
            the layers of the given module will be frozen.
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    n_max = len(children) if n is None else int(n)

    for child in children[:n_max]:
        _recursive_freeze(module=child, train_bn=train_bn)

    for child in children[n_max:]:
        _make_trainable(module=child)


def filter_params(module: Module,
                  train_bn: bool = True) -> Generator:
    """Yields the trainable parameters of a given module.
    Args:
        module: A given module
        train_bn: If True, leave the BatchNorm layers in training mode
    Returns:
        Generator
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                if param.requires_grad:
                    yield param
    else:
        for child in children:
            for param in filter_params(module=child, train_bn=train_bn):
                yield param


def _unfreeze_and_add_param_group(module: Module,
                                  optimizer: Optimizer,
                                  lr: Optional[float] = None,
                                  train_bn: bool = True):
    """Unfreezes a module and adds its parameters to an optimizer."""
    _make_trainable(module)
    params_lr = optimizer.param_groups[0]['lr'] if lr is None else float(lr)
    optimizer.add_param_group(
        {'params': filter_params(module=module, train_bn=train_bn),
         'lr': params_lr / 10.,
         })

class Cars(LightningModule):

    def __init__(self, 
                backbone: str = 'resnet50',
                train_bn: bool = True,
                batch_size: int = 8,
                lr: float = 1e-2,
                lr_scheduler_gamma: float = 1e-1,
                num_workers: int = 2) -> None
        super().__init__()
        self.backbone = backbone
        self.train_bn = train_bn
        self.batch_size = batch_size
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers

        self.__build_model()
        
    def __build_model(self):
        num_target_classes = 196
        model_func = getattr(models, self.backbone)
        backbone = model_func(pretrained=True, num_classes = num_target_classes)
    
        _layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*_layers)
        freeze(module=self.feature_extractor, train_bn=self.train_bn)

        _fc_layers = [nn.Linear(2048, num_target_classes)]
        self.fc = nn.Sequential(*_fc_layers)

    def forward(self, x):

        x = self.feature_extractor(x)
        x = x.squeeze(-1).squeeze(-1)

        x = self.fc(x)

        return x

    def training_step(self, batch, batch_idx):

        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        y_true = y.view((-1, 1)).type_as(x)
        y_bin = torch.ge(y_logits, 0)

        # 2. Compute loss & accuracy:
        train_loss = F.cross_entropy(y_logits, y_true)
        num_correct = torch.eq(y_bin.view(-1), y_true.view(-1)).sum()

        # 3. Outputs:
        tqdm_dict = {'train_loss': train_loss}
        output = OrderedDict({'loss': train_loss,
                              'num_correct': num_correct,
                              'log': tqdm_dict,
                              'progress_bar': tqdm_dict})

        return output

    def training_epoch_end(self, outputs):
        """Compute and log training loss and accuracy at the epoch level."""

        train_loss_mean = torch.stack([output['loss']
                                       for output in outputs]).mean()
        train_acc_mean = torch.stack([output['num_correct']
                                      for output in outputs]).sum().float()
        train_acc_mean /= (len(outputs) * self.batch_size)
        tensorboard_logs = {'train_loss': train_loss_mean, 'train_acc': train_acc_mean}
        return {'train_loss': train_loss_mean,  'log': tensorboard_logs}
        #return {'log': {'train_loss': train_loss_mean,
        #                'train_acc': train_acc_mean,
        #                'step': self.current_epoch}}

    def validation_step(self, batch, batch_idx):

        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        y_true = y.view((-1, 1)).type_as(x)
        y_bin = torch.ge(y_logits, 0)

        # 2. Compute loss & accuracy:
        val_loss = F.cross_entropy(y_logits, y_true)
        num_correct = torch.eq(y_bin.view(-1), y_true.view(-1)).sum()

        return {'val_loss': val_loss,
                'num_correct': num_correct}

    def validation_epoch_end(self, outputs):
        """Compute and log validation loss and accuracy at the epoch level."""

        val_loss_mean = torch.stack([output['val_loss']
                                     for output in outputs]).mean()
        val_acc_mean = torch.stack([output['num_correct']
                                    for output in outputs]).sum().float()
        val_acc_mean /= (len(outputs) * self.batch_size)
        tensorboard_logs = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        return {'val_loss': val_loss_mean,  'log': tensorboard_logs}
        #return {'log': {'val_loss': val_loss_mean,
        #                'val_acc': val_acc_mean,
        #                'step': self.current_epoch}}
        

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      self.parameters()),
                               lr=self.lr)

        #scheduler = MultiStepLR(optimizer,
        #                        milestones=self.milestones,
        #                        gamma=self.lr_scheduler_gamma)

        return optimizer#], [scheduler]

    def setup(self, stage: str):
        data_dir = '../input/stanford-car-dataset-by-classes-folder/car_data/car_data'

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        train_transforms = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.RandomCrop(400, padding=20, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std, inplace=True)
        ])
        train = ImageFolder(data_dir+'/train', train_transforms)

        # transform val
        val_transforms = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std, inplace=True)
        ])
        val = ImageFolder(data_dir+'/test', val_transforms)
        valid, _ = random_split(val, [len(val), 0])

        # assign to use in dataloaders
        self.train_dataset = train
        self.val_dataset = valid

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=True,
                            pin_memory=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=False,
                            pin_memory=True)
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--backbone',
                            default='resnet50',
                            type=str,
                            metavar='BK',
                            help='Name (as in ``torchvision.models``) of the feature extractor')
        parser.add_argument('--epochs',
                            default=5,
                            type=int,
                            metavar='N',
                            help='total number of epochs',
                            dest='nb_epochs')
        parser.add_argument('--batch-size',
                            default=8,
                            type=int,
                            metavar='B',
                            help='batch size',
                            dest='batch_size')
        parser.add_argument('--gpus',
                            type=int,
                            default=1,
                            help='number of gpus to use')
        parser.add_argument('--lr',
                            '--learning-rate',
                            default=1e-2,
                            type=float,
                            metavar='LR',
                            help='initial learning rate',
                            dest='lr')
        #parser.add_argument('--lr-scheduler-gamma',
        #                    default=1e-1,
        #                    type=float,
        #                    metavar='LRG',
        #                    help='Factor by which the learning rate is reduced at each milestone',
        #                    dest='lr_scheduler_gamma')
        parser.add_argument('--num-workers',
                            default=2,
                            type=int,
                            metavar='W',
                            help='number of CPU workers',
                            dest='num_workers')
        parser.add_argument('--train-bn',
                            default=True,
                            type=bool,
                            metavar='TB',
                            help='Whether the BatchNorm layers should be trainable',
                            dest='train_bn')
        #parser.add_argument('--milestones',
        #                    default=[5, 10],
        #                    type=list,
        #                    metavar='M',
        #                    help='List of two epochs milestones')
        return parser
    
