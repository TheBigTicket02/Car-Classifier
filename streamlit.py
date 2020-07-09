import streamlit as st
import os
from PIL import Image
import captum
import joblib
import st_model
import torch
import torch.nn.functional as F
from torchvision import transforms

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

def predict_image(logits, top=1):
    logits = main(path)

@st.cache
def main(path):
    model = st_model.EffNet.load_from_checkpoint('./cars-epoch=09-val_acc=0.9375.ckpt')
    model.eval()
    image = open_transform_image(path)
    output = predict_logits(image, model)
    _, pred_idx = torch.topk(output, 1)
    return pred_idx[0]
 

st.title('Image Classification')

st.sidebar.header("User Input Image")

img = st.sidebar.file_uploader(label='Upload your JPG file', type=['jpg'])
if img:
    image = Image.open(img)
    st.image(image)


but = st.sidebar.button(label='Predict')
if but and img is not None:
    labels = [classes()[pr] for pr in main(img)]
    result = (f'**{labels[0]}**')
    st.sidebar.markdown(result)
elif but and img is None:
    st.info('Upload image first')


st.sidebar.header("Model Interpretation")

captum = st.sidebar.radio(
    label = 'Select interpretation algorithm',
    options=["Just Prediction", "GradientShap", "IntegratedGradients", "Occlusion"]
)
if captum == 'GradientShap' and img is not None:
    st.write('Good')