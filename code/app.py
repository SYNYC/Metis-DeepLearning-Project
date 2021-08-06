import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2

import keras
from keras.models import load_model
from scipy.spatial import distance

from skimage.transform import resize
#import time


# start on terminal
# streamlit run app.py


################
##    Model   ##
################


# Load the model
model = load_model('/Users/sabrina/Metis/Projects/6_Project_DeepLearning/code/animal_face_NNtran.h5')
# Load the cascade
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


################
##   Sounds  ##
################

# Loading Cat moew sound
audio_file_0 = open('/Users/sabrina/Metis/Projects/6_Project_DeepLearning/sounds/Cat-moew.mp3', 'rb')
audio_bytes_0 = audio_file_0.read()

audio_file_1 = open('/Users/sabrina/Metis/Projects/6_Project_DeepLearning/sounds/Dog-bark.mp3', 'rb')
audio_bytes_1 = audio_file_1.read()

audio_file_2 = open('/Users/sabrina/Metis/Projects/6_Project_DeepLearning/sounds/Tiger-roar.mp3', 'rb')
audio_bytes_2 = audio_file_2.read()

################
##   Title    ##
################

# Designing the interface

image = Image.open('/Users/sabrina/Metis/Projects/6_Project_DeepLearning/ppt/afhq.jpg')


col1, col2, col3= st.beta_columns([80,2,8])
with col1:
    st.image(image)
    st.write("")
with col2:
    #st.title('Animal Faces Image Classification App')
    #st.subheader('A simple image classification web app to predict Cat-Dog-Wildlife faces with sounds effect ON')
    #st.write('A simple image classification web app to predict Cat-Dog-Wildlife faces with sounds effect ON')
    st.write("")
with col3:
    st.write("")




st.title(" Animal Faces Image Classification App")
st.write("A deep learning web app to predict Cat-Dog-Wildlife faces with sounds effect ON")


################
##    Image   ##
################


def import_and_predict(image_data, model):

        size = (150,150)    #
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(150, 150),    interpolation=cv2.INTER_CUBIC))/255.

        img_reshape = img_resize[np.newaxis,...]

        prediction = model.predict(img_reshape)

        return prediction

#############################


###########################
##    Facial Detection   ##  TBD
###########################



#############################
uploaded_file = st.file_uploader("⬇ Upload your animal here ⬇ ", type=['jpg','png', 'jpeg'])


if uploaded_file is not None:

    u_img = Image.open(uploaded_file)
    #st.success("Success")
    st.image(u_img, 'Uploaded Image', use_column_width=True)
    #show.image(u_img, 'Uploaded Image', use_column_width=True)
    #st.image(image, use_column_width=True)

    prediction = import_and_predict(u_img, model)

    if np.argmax(prediction) == 0:
        #st.subheader("It is a Cat! Meow")
        cat = '<p style="color:Blue; font-size: 30px;">It is a Cat! Meow</p>'
        st.markdown(cat, unsafe_allow_html=True)
        st.write("🔊 click sounds effect below 🐱")
        st.audio(audio_bytes_0)
    elif np.argmax(prediction) == 1:
        dog = '<p style="color:Blue; font-size: 30px;">It is a Dog! Woof</p>'
        st.markdown(dog, unsafe_allow_html=True)
        st.write("🔊 click sounds effect below 🐶")
        st.audio(audio_bytes_1)
    else:
        wildlife = '<p style="color:Blue; font-size: 30px;">It is a Wildlife! </p>'
        st.markdown(wildlife, unsafe_allow_html=True)
        st.write("Could be a Tiger🐯 Lion🦁 Fox🦊 Cheetah🐆   🔊 click sounds effect below 🔊")
        st.audio(audio_bytes_2)
        #st.subheader("None of these")


    #st.text("Probability (0: Cat, 1: Dog, 2: Wildlife)")
    prob = '<p style=" color:Black; font-size: 20px;">Probability </p>'  # (0: Cat, 1: Dog, 2: Wildlife)  #font-family:sans-serif;
    st.markdown(prob, unsafe_allow_html=True)

    st.write(("Cat     "), round(prediction[0][0], 7))
    st.write(("Dog     "),round(prediction[0][1], 7))
    st.write(("Wildlife"),round(prediction[0][2], 7))
