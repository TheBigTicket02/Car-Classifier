import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from captum.attr import Occlusion
from captum.attr import visualization as viz
import joblib
import torch
import torch.nn.functional as F
from torchvision import transforms

import os
from collections import OrderedDict

import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.metrics.functional.classification import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateLogger
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

    def __init__(self, 
                train_bn: bool = True,
                batch_size: int = 16,
                lr: float = 5e-4,
                num_workers: int = 4,
                factor: float = 0.5,
                **kwargs):
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
        self.net = EfficientNet.from_name('efficientnet-b5')
        #self.net = EfficientNet.from_pretrained('efficientnet-b5')
        in_features = self.net._fc.in_features

        _fc_layers = [nn.Linear(in_features, num_target_classes)] 
        self.net._fc = nn.Sequential(*_fc_layers)

    def forward(self, x):

        return self.net.forward(x)
    
    def training_step(self, batch, batch_idx):

        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)#.flatten()

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
        acc = accuracy(y_logits, y)#y_true)

        return {'val_loss': val_loss, 'val_acc': acc
               }#'num_correct': num_correct}

    def validation_epoch_end(self, outputs):
        """Compute and log validation loss and accuracy at the epoch level."""

        val_loss_mean = torch.stack([output['val_loss']
                                     for output in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        
        tensorboard_logs = {'val_loss': val_loss_mean, 'val_acc': avg_acc}
        return {'val_loss': val_loss_mean,  'log': tensorboard_logs}
        

    def configure_optimizers(self):
        
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                  self.parameters()),
                                   lr=self.lr)
        lr_scheduler = {'scheduler': ReduceLROnPlateau(optimizer, factor=self.factor, patience=2),'name': 'learning_rate'}
        return [optimizer], [lr_scheduler]
        

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

@st.cache
def classes():
    cl = joblib.load('classes.pkl')
    return cl

@st.cache
def open_transform_image(path):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # transform val
    img_transforms = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std, inplace=True)
        ])
    img = Image.open(path)
    image = img_transforms(img)
    
    return image

def predict_logits(img, model):
    # Convert to a batch of 1
    xb = img.unsqueeze(0)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    output = F.softmax(yb, dim=1)
    return output

@st.cache
def interpretation_transform(path):
    img_transforms = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()
        ])
    img = Image.open(path)
    image = img_transforms(img)
    
    return image

@st.cache
def main(path):
    model = EffNet.load_from_checkpoint('./cars-epoch=09-val_acc=0.9375.ckpt')
    model.eval()
    def predict(path, model):
        image = open_transform_image(path)
        output = predict_logits(image, model)
        _, pred_idx = torch.topk(output, 1)
        return pred_idx[0]
    pred_label_idx = predict(path, model)
    return model, pred_label_idx

@st.cache
def interpretation(model, input_img, transformed_img, pred_ix):
    occlusion = Occlusion(model)

    attributions_occ = occlusion.attribute(input_img,
                                       strides = (3, 20, 20),
                                       target=pred_ix,
                                       sliding_window_shapes=(3,30, 30),
                                       baselines=0)
    return attributions_occ

st.title('Car Model Classification')

st.sidebar.header("User Input Image")

img = st.sidebar.file_uploader(label='Upload your JPG file', type=['jpg'])
if img:
    image = Image.open(img)
    st.image(image)

    model, pred_ix = main(img)
    input_img = open_transform_image(img).unsqueeze(0)
    transformed_img = interpretation_transform(img)

    but = st.sidebar.button(label='Predict')
    if but:
        labels = [classes()[pr] for pr in pred_ix]
        result = (f'**{labels[0]}**')
        st.sidebar.markdown(result)

    st.sidebar.header("Model Interpretation Algorithm")

    captum = st.sidebar.radio(
        label = 'It may take several minutes',
        options=["Just Prediction", "Occlusion"]
    )

    if captum == 'Occlusion':
        attributions = interpretation(model, input_img, transformed_img, pred_ix)
        _ = viz.visualize_image_attr_multiple(np.transpose(attributions.squeeze().numpy(), (1,2,0)),
                                      np.transpose(transformed_img.squeeze().numpy(), (1,2,0)),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      show_colorbar=True,
                                      outlier_perc=2,
                                     )
        st.pyplot()
    
    if captum == 'Just prediction':
        st.write('Great!')