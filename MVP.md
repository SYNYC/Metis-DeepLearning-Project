# MVP - Animal Faces 🐈vs🐕vs🐯


### 1. EDA



- Train datatset: Found 14630 files belonging to 3 classes.
- Validation dataset: Found 1500 files belonging to 3 classes.
- Three classes : {'cat': 0, 'dog': 1, 'wild': 2}
<img src="https://github.com/SYNYC/6_Project_ImageClassification/blob/main/charts/eda-1.png">

- rescale the images size to 150 x 150

- distribution of the target variable: Cats has slightly more images data, but in general it has no huge class imbalance issue.

<img src="https://github.com/SYNYC/6_Project_ImageClassification/blob/main/charts/class_dist.png">

### 2. Non-DL model 

Random Forest





### 3. DL baseline model

My baseline model was using categorical CNN model and the accuracy on train set for 0.9999 and val set for 0.9700, but there still has gaps to improve for accuracy and loss.   

<img src="https://github.com/SYNYC/6_Project_ImageClassification/blob/main/charts/basemodel_2_acc.png">



#### Model Workflow details
3.1 model shaping
- input shape = (150,150,3)
- Conv2D & Maxpooling2D repeat 3 times  
	- filter: increased from 32-> 64-> 128)
	- kernel size: kept at 3
	- activation: ReLu
- Flatten into 1D vector and add Dense layers (512) as the fully connected layers 
- use Dense (3) for 3 classes prediction by using 'softmax' for activation

3.2 model compile
- loss : categorical_crossentropy
- optimizer: adam
- metrics: accuracy

3.3 model Summary


<img src="https://github.com/SYNYC/6_Project_ImageClassification/blob/main/charts/basemodel_2_summary.png">


#### Confusion Matrix

Here shows it wrongly predicted 45 samples in val dataset of 1500 samples
<img src="https://github.com/SYNYC/6_Project_ImageClassification/blob/main/charts/basemodel_2_cm.png">



### 4. Next Steps
1. to improve accuracy scores 
 - Transfer Learning (VGG16)
 - Image augmentation (increase training samples by rotating and horizontal flipping images)

2. apply model on steamlit app (if I have enough time)

