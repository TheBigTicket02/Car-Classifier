import streamlit as st
import os
from PIL import Image
import captum
#import effnet

#@st.cache
#def main(image):
#   model = effnet.EffNet.load_from_checkpoint('././cars-epoch=06-val_acc=0.9350.ckpt')
#   model.eval()
#   prediction = model(image)
#   class
#   out = st.sidebar.markdown(?)
#   return out
#st.info('This is a purely informational message')

#if button:
#   best = main(image)
# 
st.title('Image Classification')

st.sidebar.header("User Input Image")

img = []
img = st.sidebar.file_uploader(label='Upload your JPG file', type=['jpg'])
if img:
    image = Image.open(img)
    st.image(image, caption='Your Image')
    #image = image[:, :, [2, 1, 0]]

but = st.sidebar.button(label='Predict')
if but:
    st.sidebar.markdown('**BMW**')
    

#if button:
#   best = main()
#   out = best(image)

st.sidebar.header("Model Interpretation")

st.sidebar.radio(
    label = 'Select interpretation algorithm',
    options=["Just Prediction", "GradientShap", "IntegratedGradients", "NoiseTunnel", "Occlusion"]
)
