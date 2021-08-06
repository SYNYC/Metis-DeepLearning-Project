# Animal Faces 🐱 vs 🐶 vs 🐯

Sabrina Yang


## Abstract

Images are one kind of non-tabular data that we probably use the most often use daily, such as shopping online, taking photos, watching videos, or uploading selfies on social media, etc. Deep Learning is a great tool to let us decode the images and analyze them deeper to build an algorithm from images data.

The goal of this project is to use deep learning to do image classification on animal faces and deploy the model by building a web application for use. The main function is to identify the animals by images instead of human eye judgment, so the pet shops, animal shelters, and zoos can benefit from it.  It can also potentially help the fully self-driving car system to identify the cats, dogs, or other animals by the camera taking photo images to avoid animal run-over accidents on the road.


## Design
Neural Network can take in an input image, and assign importance (learnable weights and biases) to various aspects/objects in the image, and be able to differentiate one from the other. I'll build and train my model by taking the following steps:

- Baseline
- Convolutional Neural Net
- Transfer Learning
- Image Augmentation


## Data

This dataset is downloaded from [Kaggle](https://kaggle.com), also known as Animal Faces-HQ (AFHQ), consists of 16,130 high-quality images at 512×512 resolution.
There are three domains of classes, each providing about 5000 images. By having multiple (three) domains and diverse images of various breeds per each domain, AFHQ sets a challenging image-to-image translation problem. The classes are:

Cat;
Dog;
Wildlife (ex: tiger, lion, cheetah, fox)


- Train dataset: 14630 images
- Validation dataset: 1500 images


## Methodology

*Data Directories Setup*

setup folders with notebook by using _os.path_ and _glob_ to get data from path folders


*Images Preprocessing*

using _keras.preprocessing_ & _ImageDataGenerator_


*Workflow*
<img src="https://github.com/SYNYC/4_Project_Loan_Repayment/blob/main/charts/new/xgb5/score_0.589.png" width = "450" height = "300" >



## Tools


1. **Numpy & Pandas**: data manipulation  
2. **keras** :
- image preprocessing
- build the CNN and fully connected layers
- Transfter Learning package
3. **streamlit**: web app deployment





## Findings

### Final model

##### 0. Train/Val accuracy score vs Epochs
<img src="https://github.com/SYNYC/4_Project_Loan_Repayment/blob/main/charts/new/xgb5/score_0.589.png" width = "450" height = "300" >

##### 1. Classification report
accuracy 0.9933



              precision    recall  f1-score   support

           0       0.99      1.00      0.99       500
           1       0.99      0.98      0.99       500
           2       0.99      0.99      0.99       500

    accuracy                           0.99      1500
   macro avg       0.99      0.99      0.99      1500
weighted avg       0.99      0.99      0.99      1500

##### 2. Confusion Matrix
<img src="https://github.com/SYNYC/4_Project_Loan_Repayment/blob/main/charts/new/xgb5/score_0.589.png" width = "450" height = "300" >

get 17 wrong predictions out of 1500 samples in Validiation set


### web app


<img src="https://github.com/SYNYC/4_Project_Loan_Repayment/blob/main/charts/new/xgb5/score_0.589.png" width = "450" height = "300" >
