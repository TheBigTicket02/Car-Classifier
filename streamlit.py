import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from captum.attr import GradientShap, Occlusion, DeepLift
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
import joblib
import torch
import torch.nn.functional as F
from torchvision import transforms
import st_model

@st.cache
def classes():
    cl = joblib.load('classes.pkl')
    return cl

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

def interpretation_transform(path):
    img_transforms = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()
        ])
    img = Image.open(path)
    image = img_transforms(img)
    
    return image

def interpretation_show(attributions):
    return np.transpose(attributions.squeeze().cpu().detach().numpy(), (1,2,0))

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

@st.cache
def interpretation_deeplift(model, input_img, pred_ix):
    dl = DeepLift(model)
    attributions_dl = dl.attribute(input_img,
                                          baselines=input_img*0,
                                          target=pred_ix)

    return attributions_dl

@st.cache
def interpretation_occlusion(model, input_img, pred_ix):
    occlusion = Occlusion(model)

    attributions_occ = occlusion.attribute(input_img,
                                       strides = (3, 8, 8),
                                       target=pred_ix,
                                       sliding_window_shapes=(3,15, 15),
                                       baselines=0)
    return attributions_occ

@st.cache
def interpretation_gradient_shap(model, input_img, pred_ix):
    gradient_shap = GradientShap(model)

    # Defining baseline distribution of images
    rand_img_dist = torch.cat([input_img * 0, input_img * 1])

    attributions_gs = gradient_shap.attribute(input_img,
                                          baselines=rand_img_dist,
                                          target=pred_ix)
    return attributions_gs

default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

st.title('Car Model Classification')

st.sidebar.header("User Input Image")

img = st.sidebar.file_uploader(label='Upload your JPG file', type=['jpg'])
if img:
    image = Image.open(img)
    st.image(image)

    model, pred_ix = main(img)
    input_img = open_transform_image(img).unsqueeze(0)
    transformed_img = interpretation_transform(img)
    labels = [classes()[pr] for pr in pred_ix]
    result = (f'**{labels[0]}**')
    
    st.sidebar.header("Model Interpretation Algorithm")

    captum = st.sidebar.radio(
        label = 'Select Algorithm',
        options=["Prediction", "GradientShap", "DeepLift", "Occlusion",]
    )

    if captum == 'Occlusion':
        st.info('It may take up to 20 minutes to run Occlusion')
        attributions = interpretation_occlusion(model, input_img, pred_ix)
        _ = viz.visualize_image_attr_multiple(interpretation_show(attributions),
                                      interpretation_show(transformed_img),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      show_colorbar=True,
                                      outlier_perc=2,
                                     )
        st.pyplot()
        _ = viz.visualize_image_attr_multiple(interpretation_show(attributions),
                                      interpretation_show(transformed_img),
                                      ["original_image", "heat_map"],
                                      ["all", "all"],
                                      show_colorbar=True,
                                      outlier_perc=2,
                                     )
        st.pyplot()

    if captum == 'GradientShap':
        attributions_gs = interpretation_gradient_shap(model, input_img, pred_ix)
        _ = viz.visualize_image_attr_multiple(interpretation_show(attributions_gs),
                                      interpretation_show(transformed_img),
                                      ["original_image", "heat_map"],
                                      ["all", "absolute_value"],
                                      cmap=default_cmap,
                                      show_colorbar=True)
        st.pyplot()
    
    if captum == "DeepLift":
        attributions_gb = interpretation_deeplift(model, input_img, pred_ix)
        _ = viz.visualize_image_attr_multiple(interpretation_show(attributions_gb),
                                      interpretation_show(transformed_img),
                                      ["original_image", "heat_map"],
                                      ["all", "all"],
                                      show_colorbar=True,
                                      outlier_perc=2,
                                     )
        st.pyplot()
    
    if captum == "Prediction":
        st.sidebar.markdown(result)
