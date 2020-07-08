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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateLogger
from pytorch_lightning.loggers import WandbLogger
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

import wandb

from collections import OrderedDict
from typing import Optional
from efficientnet_pytorch import EfficientNet

wandb.login(key='44b74d6614becfad4329893ea0144da65336bdbd')

class EffNet(LightningModule):

    def __init__(self, 
                train_bn: bool = True,
                batch_size: int = 16,
                lr: float = 5e-4,
                num_workers: int = 4,
                factor: float = 0.5,
                pct_start: Optional[float] = None,
                anneal_strategy: Optional[str] = None,
                **kwargs):
        super().__init__()
        self.train_bn = train_bn
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.factor = factor
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.save_hyperparameters()

        self.__build_model()
        
    def __build_model(self):
        num_target_classes = 196
        self.net = EfficientNet.from_pretrained('efficientnet-b5')
        in_features = self.net._fc.in_features

        _fc_layers = [nn.Linear(in_features, num_target_classes)] #add hyper
                     #nn.Linear(1024, 512), #hyper
                     #nn.Linear(512, num_target_classes)]
        self.net._fc = nn.Sequential(*_fc_layers)

    def forward(self, x):

        return self.net.forward(x)
    
    def training_step(self, batch, batch_idx):

        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)

        # 2. Compute loss & accuracy:
        train_loss = F.cross_entropy(y_logits, y)
        acc = accuracy(y_logits, y)

        # 3. Outputs:
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
        
        tensorboard_logs = {'train_loss': train_loss_mean, 'train_acc': avg_acc}
        return {'train_loss': train_loss_mean,  'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):

        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)

        # 2. Compute loss & accuracy:
        val_loss = F.cross_entropy(y_logits, y)
        acc = accuracy(y_logits, y)

        return {'val_loss': val_loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        """Compute and log validation loss and accuracy at the epoch level."""

        val_loss_mean = torch.stack([output['val_loss']
                                     for output in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        
        tensorboard_logs = {'val_loss': val_loss_mean, 'val_acc': avg_acc}
        return {'val_loss': val_loss_mean,  'log': tensorboard_logs}
        

    def configure_optimizers(self):
        if (self.pct_start == None) and (self.anneal_strategy == None): #fix or
        
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      self.parameters()),
                                   lr=self.lr)
            lr_scheduler = {'scheduler': ReduceLROnPlateau(optimizer, factor=self.factor, patience=2),'name': 'learning_rate'}
            return [optimizer], [lr_scheduler]
        
        else:
            
            optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      self.parameters()),
                                   lr=self.lr)
            
            scheduler = OneCycleLR(optimizer,
                            max_lr=self.lr,
                            #steps = self.steps,
                            epochs=15, steps_per_epoch=1,
                            pct_start=self.pct_start, anneal_strategy=self.anneal_strategy)

        return [optimizer], [scheduler]

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


hyper = dict(
    train_bn = True,
    gpus=1,
    batch_size = 16,
    lr = 5e-4,
    factor = 0.5,
    num_workers = 4
    #pct_start = 0.3,
    #anneal_strategy = 'cos'
)

def main():
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    seed_everything(42)
    model = EffNet()

    # ------------------------
    # 2 SET WANDB LOGGER
    # ------------------------

    # Sweep parameters
    
    wandb_logger = WandbLogger(name='Eff1', project="Cars")

    checkpoint_cb = ModelCheckpoint(filepath = './cars-{epoch:02d}-{val_acc:.4f}',monitor='val_acc', mode='max')
    early = EarlyStopping(patience=3, monitor='val_acc', mode='max')
    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        gpus=1,
        logger=wandb_logger,
        max_epochs=15,
        progress_bar_refresh_rate=10,
        deterministic=True,
        precision=16,
        checkpoint_callback=checkpoint_cb,
        early_stop_callback=early,
        callbacks=[LearningRateLogger()],
    )

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    trainer.fit(model)
    
    wandb.save(checkpoint_cb.best_model_path)

    model.unfreeze()
    model.lr = 5e-6
    model.factor = 0.4

    wandb_logger = WandbLogger(name='Eff4', project="Cars")
    
    checkpoint_cb = ModelCheckpoint(filepath = './cars-{epoch:02d}-{val_acc:.4f}',monitor='val_acc', mode='max')
    early = EarlyStopping(patience=4, monitor='val_acc', mode='max')
    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    trainer = Trainer(
    gpus=1,
    logger=wandb_logger,
    max_epochs=25,
    progress_bar_refresh_rate=5,
    deterministic=True,
    precision=16,
    checkpoint_callback=checkpoint_cb,
    early_stop_callback=early,
    callbacks=[LearningRateLogger()],
    )

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    trainer.fit(model)
    
    wandb.save(checkpoint_cb.best_model_path)

if __name__ == '__main__':
    main()