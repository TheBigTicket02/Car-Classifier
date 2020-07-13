import os
from argparse import ArgumentParser, Namespace
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.metrics.functional.classification import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateLogger
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from efficientnet_pytorch import EfficientNet

import wandb

from collections import OrderedDict
from typing import Optional
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


class EffNet(LightningModule):

    def __init__(self, 
                num_target_classes = 196,
                backbone: str = 'efficientnet-b5',
                hidden_1: int = 1024,
                hidden_2: int = 512,
                dropout: float = 0.3, 
                train_bn: bool = True,
                milestones: tuple = (5, 10),
                batch_size: int = 20,
                lr: float = 1e-3,
                lr_scheduler_gamma: float = 2e-1,
                wd: float = 1e-6,
                factor: float = 0.5,
                num_workers: int = 4,
                **kwargs):
        super().__init__()
        self.num_target_classes = num_target_classes
        self.backbone= backbone
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.dropout = dropout
        self.batch_size = batch_size
        self.train_bn = train_bn
        self.milestones = milestones
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.wd = wd
        self.num_workers = num_workers
        self.factor = factor
        self.save_hyperparameters()

        self.__build_model()
        
    def __build_model(self):
        self.net = EfficientNet.from_pretrained(self.backbone)
        
        _layers = list(self.net.children())[:1]
        self.feature_extractor = nn.Sequential(*_layers)
        freeze(module=self.feature_extractor, train_bn=self.train_bn)

        in_features = self.net._fc.in_features
        _fc_layers = [nn.Linear(in_features, self.hidden_1),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_1, self.hidden_2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_2, self.num_target_classes)]
        self.net._fc = nn.Sequential(*_fc_layers)

    def forward(self, x):
        return self.net.forward(x)
    
    def train(self, mode=True):
        super().train(mode=mode)

        epoch = self.current_epoch
        if epoch < self.milestones[0] and mode:
            # feature extractor is frozen (except for BatchNorm layers)
            freeze(module=self.feature_extractor,
                   train_bn=self.train_bn)

        elif self.milestones[0] <= epoch < self.milestones[1] and mode:
            # Unfreeze last two layers of the feature extractor
            freeze(module=self.feature_extractor,
                   n=-2,
                   train_bn=self.train_bn)

    def training_step(self, batch, batch_idx):

        x, y = batch
        y_logits = self.forward(x)

        train_loss = F.cross_entropy(y_logits, y)
        acc = accuracy(y_logits, y)

        tqdm_dict = {'train_loss': train_loss}
        output = OrderedDict({'loss': train_loss,
                               'train_acc': acc,
                              'log': tqdm_dict,
                             'progress_bar': tqdm_dict})

        return output

    def training_epoch_end(self, outputs):
        """Compute and log training loss and accuracy at the epoch level."""

        train_loss_mean = torch.stack([output['loss']
                                       for output in outputs]).mean()
        avg_acc = torch.stack([x['train_acc'] for x in outputs]).mean()
        
        tensorboard_logs = {'train_loss': train_loss_mean, 'train_acc1': avg_acc}

        return {'train_loss': train_loss_mean,  'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):

        x, y = batch
        y_logits = self.forward(x)

        val_loss = F.cross_entropy(y_logits, y)
        acc, acc2 = self.__accuracy(y_logits, y, topk=(1,2))

        return OrderedDict({'val_loss': val_loss, 'val_acc': acc,
                            'top2_acc': acc2})

    def validation_epoch_end(self, outputs):
        """Compute and log validation loss and accuracy at the epoch level."""

        val_loss_mean = torch.stack([output['val_loss']
                                     for output in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_acc2 = torch.stack([x['top2_acc'] for x in outputs]).mean()
        
        tensorboard_logs = {'val_loss': val_loss_mean, 'val_acc': avg_acc,
                            'top2_acc': avg_acc2}
        return {'val_loss': val_loss_mean,  'log': tensorboard_logs}


    @classmethod
    def __accuracy(cls, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res    

    def configure_optimizers(self):
        
        if self.current_epoch <= self.milestones[1]:

            optimizer = Adam(filter(lambda p: p.requires_grad,
                                      self.parameters()),
                               lr=self.lr)

            lr_scheduler = {'scheduler': MultiStepLR(optimizer,
                                milestones=self.milestones,
                                gamma=self.lr_scheduler_gamma),
                            'name': 'learning_rate' }

            return [optimizer], [lr_scheduler]

        else:
            optimizer = Adam(self.parameters(),
                lr=self.lr, weight_decay=self.wd)
            lr_scheduler = {'scheduler': ReduceLROnPlateau(optimizer, factor=self.factor, 
            patience=3, mode='max'),'name': 'learning_rate',
            'monitor': 'val_acc1'}
            return [optimizer], [lr_scheduler]
    

    def setup(self, stage: str):
        data_dir = '../input/stanford-car-dataset-by-classes-folder/car_data/car_data'

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        train_transforms = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.RandomCrop(400, padding=20, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std, inplace=True)
        ])
        train = ImageFolder(os.path.join(data_dir,'train'), train_transforms)

        # transform val
        val_transforms = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std, inplace=True)
        ])
        val = ImageFolder(os.path.join(data_dir,'test'), val_transforms)
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
'''
    @staticmethod
    def add_model_specific_args():
        parser = ArgumentParser()
        parser.add_argument('--num_target_classes',
                            default=196,
                            type=int,
                            metavar='NUM',
                            help='Number of target classes')
        parser.add_argument('--backbone',
                            default='efficientnet-b5',
                            type=str,
                            metavar='BK',
                            help='Name of the feature extractor')
        parser.add_argument('--epochs',
                            default=15,
                            type=int,
                            metavar='N',
                            help='total number of epochs',
                            dest='nb_epochs')
        parser.add_argument('--patience',
                            default=3,
                            type=int,
                            metavar='ES',
                            help='early stopping',
                            dest='patience')
        parser.add_argument('--batch-size',
                            default=16,
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
                            default=1e-3,
                            type=float,
                            metavar='LR',
                            help='initial learning rate',
                            dest='lr')
        parser.add_argument('--weight_decay',
                            default=0,
                            type=float,
                            metavar='WD',
                            help='L2 Penalty',
                            dest='weight_decay')
        parser.add_argument('--num-workers',
                            default=4,
                            type=int,
                            metavar='W',
                            help='number of CPU workers',
                            dest='num_workers')
        parser.add_argument('--factor',
                            default=0.5,
                            type=float,
                            metavar='FAC',
                            help='Factor by which learning rate will be reduced',
                            dest='factor')
        parser.add_argument('--use_onecycle',
                            default=False,
                            type=bool,
                            metavar='OCLR',
                            help='Enable One Cycle Learning Rate Scheduler')
        parser.add_argument('--pct_start',
                            default=0.3,
                            type=float,
                            metavar='PS',
                            help='The Percentage of the cycle spent increasing LR')
        parser.add_argument('--anneal_strategy',
                            default='cos',
                            type=str,
                            metavar='AS',
                            help='Cosine Anneling')
        return parser
'''

wandb.login(key='44b74d6614becfad4329893ea0144da65336bdbd')

def main():

    seed_everything(42)
    model = EffNet()
    
    wandb_logger = WandbLogger(name='Best1', project="Cars")

    checkpoint_cb = ModelCheckpoint(filepath = './cars-{epoch:02d}-{val_acc:.4f}',monitor='val_acc1', mode='max')
    early = EarlyStopping(patience=5, monitor='val_acc1', mode='max')

    trainer = Trainer(
        gpus=1,
        logger=wandb_logger,
        max_epochs=24,
        progress_bar_refresh_rate=10,
        deterministic=True,
        precision=16,
        checkpoint_callback=checkpoint_cb,
        early_stop_callback=early,
        callbacks=[LearningRateLogger()],
    )

    trainer.fit(model)
    
    wandb.save(checkpoint_cb.best_model_path)

#def get_args() -> Namespace:
#    parser = EffNet.add_model_specific_args()
#    return parser.parse_args()

if __name__ == '__main__':
    main()
