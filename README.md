# Natural-Image-Classification-with-Different-CNN-Models

## Description :
Image classification is the process of labeling images according to predefined categories. An image classification model is fed a set of 
images within a specific category. Based on this set, the algorithm learns which class the test images belong to, and can then predict the
correct class of future image inputs, and can even measure how accurate the predictions are. In this project, we have trained our model 
using Convolutional Neural Network on Natural Scenes around the world dataset.

## Dataset:
This is the Data of Natural Scenes around the world.
### Content: 
This Data contains around 25k images of size 150x150 distributed under 6 categories
i.e. Buildings, Forest, Glacier , Mountain, Sea and Street.
The Train, Test and Prediction data is separated in each zip files. There are around 14k images in
Train, 3k in Test and 7k in Prediction.
You are supposed to work in following manner:
• Training set = 14k+ 150x150 Images in seg_train folder for training spread.
• Validation set = 3k+ 150x150 Images in seg_test folder for cross-validation spread.
• Test set = 7k+ 150x150 Images in seg_pred folder as test spread.
#### Example: 
<img src="https://github.com/A-Janj/Natural-Image-Classification-with-Different-CNN-Models/blob/main/Images/train%20images%20WA.PNG" width="800" height="300">

### Dataset available at:
<a href="https://www.kaggle.com/puneet6060/intel-image-classification/version/2">Kaggle link for Natural scenes classification dataset</a>

### Folders explained:
The following folders contain:
1. **code :** it contains the code for the best performing CNN archi with its different hyperparameters.
2. **Report :** has the formal report of the assignment
3. **pdf code print :** pdf code prints contain the pdf of all the files of code run with different CNN acrhitectures used and their hyperparameters that I tried
4. **images :** Some visual aids are stored to represent results of the best performing model.


## Architecture
* Convolution Neural Networks => VGG16 and InceptionResNetV2

### Requirements
* Python 3.6.10  
* Numpy 1.18.4   
* Keras 2.4.3
* Matplotlib 3.2.1
* Scikit-learn 0.23.1


## Results and Visualization
* The CNN I found the best to give accuracy on prediction/test data was at 89% VGG16 without augented data and stochastic gradient descent with learning rate 0.1.

The graph of accuracy of test and validation datasets along with loss graph:

<img src="https://github.com/A-Janj/Natural-Image-Classification-with-Different-CNN-Models/blob/main/Images/both%20accuracy%20loss.PNG">

The confusion matrix/heatmap for the predicted images:

<img src="https://github.com/A-Janj/Natural-Image-Classification-with-Different-CNN-Models/blob/main/Images/heatmap.PNG">

This is the comparison table made with the different parameters tried:

<img src="https://github.com/A-Janj/Natural-Image-Classification-with-Different-CNN-Models/blob/main/Images/Comparison%20Table.png" width="450" height="300">
       
