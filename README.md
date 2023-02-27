# Medicinal-Plant-Detection
It is an image classification model build up using PyTorch, which is an optimized Deep Learning tensor library based on Python and Torch and is mainly used for applications using GPUs and CPUs.
PyTorch is favored over other Deep Learning frameworks like TensorFlow and Keras since it uses dynamic computation graphs and is completely Pythonic.

#Data Preparation:
We have developed our own dataset that contains images taken from online contents and live photos taken using mobile phones.
For the same, we have used mobile phones of various resolutions and also we have collected data from various locations of Kerala.
The reason for choosing various locations was that the same species of plants may have different color and shape.
There are two separate folders for train and val. The train folder contains 80% of each classes and the rest 20% is used for validating.
We have total of 10 categories to predict Neem tree ,Moringa tree, Indian GooseBerry, Tamarind, Guava, Curry leaves, Christmas bush , Golden Apple, Indian aloe, Mymosa
Each classes have 300 images.

#Dataset Augmentation:
We are using some of the augmentation techniques for reducing the data deficiency problem. They are RandomResizedCrop(224), RandomHorizontalFlip(), RandomRotation(270) and
transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2).

#Research Methodology:
The proposed system has used the pre-trained network ResNet101 for creating the models.
The models have been trained using the GPU, Tesla V100-PCIE-32GB, which took less than nearly 45 minutes for the entire model building process.

#Results
Using the proposed model and Transfer Learning, results revealed a 97.66 percent accuracy on the challenging task.
