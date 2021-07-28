# Project 6 Deep Learning : Cat vs Dog Image Classification 

## Question


The idea of this project is to use a deep learning algorithm to classify whether the images contain either a dog or cat.

 This images classification will be binary (cat or dog) and can help the pet shops and animal shelters to identify if the object is a cat or dog by only images instead of human eye judgement. This model can also potentially help to the full self-driving car system to identify the cats and dogs by camera taking photo images to avoid the animal runover accidents on the road. 




## Data


### Cats & Dogs

-  Resource: downloaded from [Kaggle.com](https://www.kaggle.com/c/dogs-vs-cats/data)

-  Photo images examples:


Cat
<img src="https://github.com/SYNYC/4_Project_Loan_Repayment/blob/main/charts/LoanDictionary.png">

Dog
<img src="https://github.com/SYNYC/4_Project_Loan_Repayment/blob/main/charts/LoanDictionary.png">

- it's a class balanced dataset


## Tools
1. tensonflow
2. keras
	* from resnet50 import ResNet50
	* from keras.preprocessing import image
	* from imagenet_utils import preprocess_input, decode_predictions

3. sklearn
    * from sklearn import preprocessing, model_selection, train_test_split
4. validation tools






## MVP Goal:

✨Take a subset and run baseline model for Random Forest and a single layer of Sequential keras model✨
