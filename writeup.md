# Animal Faces 🐱 vs 🐶 vs 🐯

Sabrina Yang


## Abstract

Images are one kind of non-tabular data that we probably use the most often use daily, such as shopping online, taking photos, watching videos, or uploading selfies on social media, etc. Deep Learning is a great tool to let us decode the images and analyze them deeper to build an algorithm from images data.

The goal of this project is to use deep learning to do image classification on animal faces and deploy the model by building a web application for use. The main function is to identify the animals by images instead of human eye judgment, so the pet shops, animal shelters, and zoos can benefit from it.  It can also potentially help the fully self-driving car system to identify the cats, dogs, or other animals by the camera taking photo images to avoid animal run-over accidents on the road.


## Design
Neural Network can take in an input image, and assign importance (learnable weights and biases) to various aspects/objects in the image, and be able to differentiate one from the other. I'll build and train my model by taking the following steps:

1. Baseline
2. Convolutional Neural Net
3. Transfer Learning
4. Image Augmentation


## Data

This dataset is downloaded from [Kaggle](https://kaggle.com), also known as Animal Faces-HQ (AFHQ), consists of 16,130 high-quality images at 512×512 resolution.
There are three domains of classes, each providing about 5000 images. By having multiple (three) domains and diverse images of various breeds per each domain, AFHQ sets a challenging image-to-image translation problem. The classes are:

Cat;
Dog;
Wildlife (ex: tiger, lion, cheetah, fox)


- Train dataset: 14630 images
- Validation dataset: 1500 images


## Methodology

**1. Data Directories Setup**

setup folders with notebook by using _os.path_ and _glob_ to get data from path folders


**2. Images Preprocessing**

using _keras.preprocessing_ & _ImageDataGenerator_


**3. Workflow**
<img src="https://github.com/SYNYC/6_Project_ImageClassification/blob/main/charts/workflow.png" >



## Tools


1. **Numpy & Pandas**: data manipulation  
2. **keras** :
- image preprocessing
- build the CNN and fully connected layers
- Transfter Learning package
3. **streamlit**: web app deployment





## Findings

### 1. Final model

##### a. Train/Val accuracy score vs Epochs
<img src="https://github.com/SYNYC/6_Project_ImageClassification/blob/main/charts/accuracy_loss.png" >

##### b. Classification report
accuracy 0.9933



              precision    recall  f1-score   support

           0       0.99      1.00      0.99       500
           1       0.99      0.98      0.99       500
           2       0.99      0.99      0.99       500

    accuracy                           0.99      1500
   macro avg       0.99      0.99      0.99      1500
weighted avg       0.99      0.99      0.99      1500

##### c. Confusion Matrix
got 17 wrong predictions out of 1500 samples in Validation dataset
<img src="https://github.com/SYNYC/6_Project_ImageClassification/blob/main/charts/cm.png" width = "450" height = "300" >




### 2. Web app 
- enable to upload image to test model predictions
- included animal sounds effect
- included Probability Chart of predicting images between cat, dog and wildlife

**EXAMPLE 1**
<img src="https://github.com/SYNYC/6_Project_ImageClassification/blob/main/charts/app_cat.png" >

**EXAMPLE 2**
<img src="https://github.com/SYNYC/6_Project_ImageClassification/blob/main/charts/app_dog_.png" >

**EXAMPLE 3**
<img src="https://github.com/SYNYC/6_Project_ImageClassification/blob/main/charts/app_wild.png" >
