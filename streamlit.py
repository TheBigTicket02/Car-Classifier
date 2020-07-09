import streamlit as st
from PIL import Image
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import torchvision
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import visualization as viz
import joblib
import st_model
import torch
import torch.nn.functional as F
from torchvision import transforms

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
    model = st_model.EffNet.load_from_checkpoint('./cars-epoch=09-val_acc=0.9375.ckpt')
    model.eval()
    def predict(path, model):
        image = open_transform_image(path)
        output = predict_logits(image, model)
        _, pred_idx = torch.topk(output, 1)
        return pred_idx[0]
    pred_label_idx = predict(path, model)
    return model, pred_label_idx
 

st.title('Image Classification')

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
        occlusion = Occlusion(model)

        attributions_occ = occlusion.attribute(input_img,
                                       strides = (3, 50, 50),
                                       target=pred_ix,
                                       sliding_window_shapes=(3,60, 60),
                                       baselines=0)
        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().numpy(), (1,2,0)),
                                      np.transpose(transformed_img.squeeze().numpy(), (1,2,0)),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      show_colorbar=True,
                                      outlier_perc=2,
                                     )
        st.pyplot()




