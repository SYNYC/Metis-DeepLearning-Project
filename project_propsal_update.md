# Project Proposal - Deep Learning : Animal Faces 🐱 vs 🐶 vs 🐯

### Question/need


The idea of this project is to use a deep learning algorithm to identify the images is a cat, dog or wildlife.


 This images classification will be categorical in 3 classes : 
 1. Cat 
 2. Dog 
 3. Wildlife (ex: tiger, lion)


Use cases:
1. to identify the object by only images instead of human eye judgement
- pet shops
- animal shelters 
- zoos

2. to help full self-driving car system to identify the cats, dogs or other animals by camera taking photo images 
to avoid the animal run-over accidents on the road


### Data


##### Cat & Dog & Wildlife

-  Resource: downloaded from [Kaggle.com](hhttps://www.kaggle.com/andrewmvd/animal-faces)

- Pre-split into train and validation 2 folders

- In each folders, images are labeled with the animals types in its types folders


### Tools
0. load images data & create directory
   * import pathlib
   * from glob import glob


1. Keras (Deep Learning)
    * import keras
	* from keras.preprocessing.image import ImageDataGenerator
	* from keras.models import Sequential
	* from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, InputLayer, Dropout 

2. sklearn
    - confusion matrix

3. Streamlit




### MVP Goal:

✨Take a subset and run baseline model for Random Forest and a baseline Deep Learning model✨
