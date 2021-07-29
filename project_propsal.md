# Project 6 Deep Learning : Cat vs Dog Image Classification 

## Question


The idea of this project is to use a deep learning algorithm to classify whether the images contain either a dog or cat.

 This images classification will be binary (cat or dog) and can help the pet shops and animal shelters to identify if the object is a cat or dog by only images instead of human eye judgement. This model can also potentially help to the full self-driving car system to identify the cats and dogs by camera taking photo images to avoid the animal runover accidents on the road. 




## Data


### Cats & Dogs

-  Resource: downloaded from [Kaggle.com](https://www.kaggle.com/c/dogs-vs-cats/data)

-  Data details:
	1. Train & Test pre-split: From downloaded file, it's already split into 2 seperate folders named as Test and Train.

	2. Size: 
	- Test folder contains 12500 images with mixed of cats and dogs but without label; 
	- Train folder contains 10000 images with cat label and another 10000 images with dog label. It's also a class balanced dataset. 
 
-  Photo images examples:


Cat
<img src="https://github.com/SYNYC/4_Project_Loan_Repayment/blob/main/charts/LoanDictionary.png">

Dog
<img src="https://github.com/SYNYC/4_Project_Loan_Repayment/blob/main/charts/LoanDictionary.png">




## Tools
1. tensonflow
2. keras
	* from resnet50 import ResNet50
	* from keras.preprocessing import image
	* from imagenet_utils import preprocess_input, decode_predictions

3. sklearn
    * from sklearn import preprocessing, model_selection
    
4. validation tools






## MVP Goal:

✨Take a subset and run baseline model for Random Forest and a single layer of Sequential keras model✨
