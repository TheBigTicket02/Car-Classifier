import os
from typing import Optional
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.metrics.functional.classification import accuracy

from pytorch_lightning.loggers import WandbLogger
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

from efficientnet_pytorch import EfficientNet


class EffNet(LightningModule):
    """
    Untrained Efficient Net(b5)
    """

    def __init__(
        self,
        train_bn: bool = True,
        batch_size: int = 16,
        lr: float = 5e-4,
        num_workers: int = 4,
        factor: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.train_bn = train_bn
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.factor = factor
        self.save_hyperparameters()

        self.__build_model()

    def __build_model(self):
        num_target_classes = 196
        self.net = EfficientNet.from_name("efficientnet-b5")
        # self.net = EfficientNet.from_pretrained('efficientnet-b5')
        in_features = self.net._fc.in_features

        _fc_layers = [nn.Linear(in_features, num_target_classes)]
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


    def validation_step(self, batch, batch_idx):

        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)

        # 2. Compute loss & accuracy:
        val_loss = F.cross_entropy(y_logits, y)
        acc = accuracy(y_logits, y)  # y_true)

        return {"val_loss": val_loss, "val_acc": acc}  # 'num_correct': num_correct}

    def validation_epoch_end(self, outputs):
        """Compute and log validation loss and accuracy at the epoch level."""

        val_loss_mean = torch.stack([output["val_loss"] for output in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

        tensorboard_logs = {"val_loss": val_loss_mean, "val_acc": avg_acc}
        return {"val_loss": val_loss_mean, "log": tensorboard_logs}

    def configure_optimizers(self):

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, factor=self.factor, patience=2),
            "name": "learning_rate",
        }
        return [optimizer], [lr_scheduler]

    def setup(self, stage: str):
        data_dir = "../input/stanford-car-dataset-by-classes-folder/car_data/car_data"

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        train_transforms = transforms.Compose(
            [
                transforms.Resize((400, 400)),
                transforms.RandomCrop(350),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std, inplace=True),
            ]
        )
        train = ImageFolder(data_dir + "/train", train_transforms)

        # transform val
        val_transforms = transforms.Compose(
            [
                transforms.Resize((400, 400)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std, inplace=True),
            ]
        )
        val = ImageFolder(data_dir + "/test", val_transforms)
        valid, _ = random_split(val, [len(val), 0])

        # assign to use in dataloaders
        self.train_dataset = train
        self.val_dataset = valid

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
