import streamlit as st
import cv2
from PIL import Image
from Screen_extraction import screen_extraction
from Number_detection import number_detection
from Number_classification import final_inference
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import cv2

import torch
from torch import nn, optim
import torchvision.transforms as T
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

import timm
import segmentation_models_pytorch as smp
import imutils
from skimage.transform import ProjectiveTransform

import os
from tqdm import tqdm
from PIL import Image
import albumentations as A
from sklearn.model_selection import train_test_split
import gc
import glob

import random
import yolov5
import paddleocr
from paddleocr import PaddleOCR,draw_ocr
from ensemble_boxes import *
import re
import torch.nn.functional as F
import copy


root = '/Users/aryanlath/Downloads/CloudPhysician-s-Vital-Extraction-Challenge-main/Saved Models'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Corner Regression Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('resnet34', pretrained=True, num_classes=768)
        self.ll = nn.Linear(768,8)
    
    def forward(self, img):
        x = self.model(img)
        x = self.ll(x)

        return x

model_reg = CNN()

## UNET++ Model
ENCODER = 'resnext101_32x8d'
ENCODER_WEIGHTS = 'imagenet'

model_unet = smp.UnetPlusPlus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=1,
    in_channels=3,
    activation=None,
)

model_reg = model_reg.to(device)
model_unet = model_unet.to(device)

model_reg.load_state_dict(torch.load(root + 'corner_reg.pt', map_location=device))
model_unet.load_state_dict(torch.load(root + 'unet++_weights.pt', map_location=device))

preprocessing_unet = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

def vital_extraction(img_path, mode = 'accurate'):

  if mode == 'accurate':

    img = cv2.imread(img_path)
    transformed_image = screen_extraction(img_path, model_unet, model_reg, preprocessing_unet, mode)

    detection_dict = number_detection(transformed_image, mode)

    output_dict = final_inference(detection_dict)

    return output_dict

  else:
    
    img = cv2.imread(img_path)

    transformed_image = screen_extraction(img_path, model_unet, model_reg, preprocessing_unet, mode)

    detection_dict = number_detection(transformed_image, mode)

    return detection_dict

def main():
    st.title("Streamlit Tutorial")
    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Vitals Extraction WebApp</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.image(our_image)
    if st.button("Recognise"):
        result= vital_extraction(image_file)
        st.success('The vitals are {}'.format(result))

if __name__ == '__main__':
    main()
 
