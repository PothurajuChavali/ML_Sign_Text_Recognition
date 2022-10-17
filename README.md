# ML_Sign_Text_Recognition

Character recognition on how to extract characters/digits from the sign text images

Datasets:
EMNIST dataset - Balanced dataset
Extended MNIST dataset(EMNIST) whis a set of handwritten character digits derived from the NIST Special Database 19 and 
converted to a 28x28 pixel image format and dataset structure that directly matches the MNIST dataset. 
Further information on the dataset contents and conversion process can be found in the paper available at https://arxiv.org/abs/1702.05373v1.

Balanced dataset
The EMNIST Balanced dataset is meant to address the balance issues in the ByClass and ByMerge datasets. 
It is derived from the ByMerge dataset to reduce mis-classification errors due to capital and lower case letters and also has an equal number of samples per class. This dataset is meant to be the most applicable.

train: 112,800
test: 18,800
total: 131,600
classes: 47 (balanced)

The trained model based on SVM classifier is saved and available.

How to run predictions on new/test images:
1) To use the existing model, you can directly unzip the saved model(signtext_recognition_model_balanced_dataset.pkl) and use it.
2) There is a mapping file(emnist-balanced-mapping.txt) for EMNIST balanced dataset(maps the class to corresponding ascii value) already added in the repo
3) Add the image path on which we need to recognize the text data in line number 80 in main.py
 
