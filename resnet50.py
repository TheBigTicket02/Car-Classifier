import os
from argparse import ArgumentParser, Namespace
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.metrics.functional.classification import accuracy
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateLogger,
)
from pytorch_lightning.loggers import WandbLogger
from torch import optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

import wandb

from collections import OrderedDict
from typing import Optional, Generator, Union
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


def _recursive_freeze(module: Module, train_bn: bool = True) -> None:
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


def freeze(module: Module, n: Optional[int] = None, train_bn: bool = True) -> None:
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


wandb.login(key="44b74d6614becfad4329893ea0144da65336bdbd")


class ResNet50(LightningModule):
    def __init__(
        self,
        train_bn: bool = True,
        batch_size: int = 70,
        lr: float = 1e-3,
        num_workers: int = 4,
        hidden_1: int = 1024,
        hidden_2: int = 512,
        epoch_freeze: int = 8,
        pct_start: float = 0.2,
        anneal_strategy: str = "cos",
        **kwargs
    ):
        super().__init__()
        self.train_bn = train_bn
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.epoch_freeze = epoch_freeze
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.save_hyperparameters()

        self.__build_model()

    def __build_model(self):
        num_target_classes = 196
        backbone = models.resnet50(pretrained=True)

        _layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*_layers)
        freeze(module=self.feature_extractor, train_bn=self.train_bn)

        _fc_layers = [
            nn.Linear(2048, self.hidden_1),
            nn.Linear(self.hidden_1, self.hidden_2),
            nn.Linear(self.hidden_2, num_target_classes),
        ]
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

        # 2. Compute loss & accuracy:
        train_loss = F.cross_entropy(y_logits, y)
        acc = accuracy(y_logits, y)

        # 3. Outputs:
        tqdm_dict = {"train_loss": train_loss}
        output = OrderedDict(
            {
                "loss": train_loss,
                "train_acc": acc,
                "log": tqdm_dict,
                "progress_bar": tqdm_dict,
            }
        )

        return output

    def training_epoch_end(self, outputs):
        """Compute and log training loss and accuracy at the epoch level."""

        train_loss_mean = torch.stack([output["loss"] for output in outputs]).mean()
        avg_acc = torch.stack([x["train_acc"] for x in outputs]).mean()

        tensorboard_logs = {"train_loss": train_loss_mean, "train_acc": avg_acc}
        return {"train_loss": train_loss_mean, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):

        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)

        # 2. Compute loss & accuracy:
        val_loss = F.cross_entropy(y_logits, y)
        acc = accuracy(y_logits, y)

        return {"val_loss": val_loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        """Compute and log validation loss and accuracy at the epoch level."""

        val_loss_mean = torch.stack([output["val_loss"] for output in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()

        tensorboard_logs = {"val_loss": val_loss_mean, "val_acc": avg_acc}
        return {"val_loss": val_loss_mean, "log": tensorboard_logs}

    def configure_optimizers(self):
        if self.current_epoch < self.epoch_freeze:

            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
            )
            return optimizer

        else:

            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
            )

            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=15,
                pct_start=self.pct_start,
                anneal_strategy=self.anneal_strategy,
            )

        return [optimizer], [scheduler]

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


def main():
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    seed_everything(42)
    model = ResNet50()

    # ------------------------
    # 2 SET WANDB LOGGER
    # ------------------------
    wandb.login(key=os.environ.get("WANDB_API_KEY"))

    wandb_logger = WandbLogger(name="Test", project="Cars")

    checkpoint_cb = ModelCheckpoint(
        filepath="./cars-{epoch:02d}-{val_acc:.4f}", monitor="val_acc", mode="max"
    )
    early = EarlyStopping(patience=3, monitor="val_acc", mode="max")

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

    trainer.fit(model)

    wandb.save(checkpoint_cb.best_model_path)


if __name__ == "__main__":
    main()
