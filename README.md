# Tuberculosis Detection from Chest X-Rays using Deep Learning

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify chest X-ray images for the presence of tuberculosis. The model is trained on the "Tuberculosis Chest X-rays" dataset from Kaggle.

## Table of Contents
1.  [Overview](#overview)
2.  [Dataset](#dataset)
3.  [Workflow](#workflow)
4.  [Model Architecture](#model-architecture)
5.  [Setup and Installation](#setup-and-installation)
6.  [How to Run](#how-to-run)
7.  [Results](#results)
8.  [Files in this Repository](#files-in-this-repository)
9.  [Future Work](#future-work)

## Overview
The primary goal of this project is to build and train a deep learning model capable of distinguishing between chest X-rays of healthy individuals and those indicative of tuberculosis. This involves data acquisition, preprocessing, model training, evaluation, and saving the model for inference.


## Workflow
1.  **Dataset Download:**
    * The Kaggle API is used to download the dataset directly into the Colab environment. [cite: 1]
    * The downloaded ZIP file is then extracted. [cite: 1]
2.  **Preprocessing:**
    * The `shenzhen_metadata.csv` file is read to identify 'normal' and 'tuberculosis' (positive) cases. [cite: 3, 6]
    * Images are moved into respective `data/normal` and `data/positive` directories based on the metadata. [cite: 6, 7]
3.  **Data Splitting:**
    * The `splitfolders` library is used to partition the data into training and testing sets. [cite: 1, 2]
    * The split ratio used is 80% for training and 20% for testing (`ratio=(.8, .0, 0.2)`). The test set is used for validation during training. [cite: 8]
4.  **Image Augmentation and Loading:**
    * `ImageDataGenerator` from Keras is used for:
        * Rescaling pixel values (dividing by 255.0). [cite: 3]
        * Resizing images to a target size of 150x150 pixels. [cite: 3]
        * Generating batches of image data for training (`train_data_gen`) and testing/validation (`test_data_gen`). [cite: 3]
5.  **Model Building:**
    * A Sequential CNN model is constructed with several convolutional, max-pooling, flatten, dense, and dropout layers. [cite: 11]
6.  **Training:**
    * The model is compiled using the 'Adam' optimizer, 'binary_crossentropy' loss function, and 'accuracy' as the metric. [cite: 11]
    * Training is performed for 20 epochs. [cite: 11, 12]
7.  **Analysis/Evaluation:**
    * Training and validation loss and accuracy are plotted over epochs to observe model performance. [cite: 15, 17]
8.  **Model Saving:**
    * The trained model is saved in HDF5 format (`Tuberculosis.h5`). [cite: 18]
    * The model is also converted to TensorFlow Lite format (`Tuberculosis.tflite`). [cite: 18]
9.  **Prediction/Inference:**
    * A function `predict_image` loads the saved H5 model and predicts the class (Normal or Tuberculosis) for a given input image, displaying the prediction and confidence. [cite: 21, 22, 23, 24, 25]

## Model Architecture
The model is a Sequential CNN with the following structure:

| Layer (type)           | Output Shape          | Param #   |
| :--------------------- | :-------------------- | :-------- |
| conv2d (Conv2D)        | (None, 150, 150, 32)  | 896       |
| max_pooling2d (MaxPool) | (None, 75, 75, 32)    | 0         |
| conv2d_1 (Conv2D)      | (None, 75, 75, 64)    | 18,496    |
| max_pooling2d_1 (MaxPool)| (None, 37, 37, 64)    | 0         |
| conv2d_2 (Conv2D)      | (None, 37, 37, 128)   | 73,856    |
| max_pooling2d_2 (MaxPool)| (None, 18, 18, 128)   | 0         |
| conv2d_3 (Conv2D)      | (None, 18, 18, 192)   | 221,376   |
| max_pooling2d_3 (MaxPool)| (None, 9, 9, 192)     | 0         |
| flatten (Flatten)      | (None, 15552)         | 0         |
| dense (Dense)          | (None, 128)           | 1,990,784 |
| dropout (Dropout)      | (None, 128)           | 0         |
| dense_1 (Dense)        | (None, 228)           | 29,412    |
| dropout_1 (Dropout)    | (None, 228)           | 0         |
| dense_2 (Dense)        | (None, 270)           | 61,830    |
| dropout_2 (Dropout)    | (None, 270)           | 0         |
| dense_3 (Dense)        | (None, 1)             | 271       |

**Total params:** 2,396,923 [cite: 28]
**Trainable params:** 2,396,921 [cite: 28]
**Non-trainable params:** 0 [cite: 28]

## Setup and Installation
1.  **Prerequisites:**
    * Python 3
    * Jupyter Notebook or Google Colab environment
2.  **Libraries:**
    You can install the necessary libraries using pip:
    ```bash
    pip install tensorflow numpy pandas matplotlib opencv-python split-folders kaggle
    ```
    It's recommended to use a `requirements.txt` file for managing dependencies.
3.  **Kaggle API Token:**
    * To download the dataset, you'll need your Kaggle API token (`kaggle.json`). [cite: 1]
    * Download it from your Kaggle account page (`Account` -> `API` -> `Create New API Token`).
    * When running the notebook, you will be prompted to upload this `kaggle.json` file. [cite: 1]

## How to Run
1.  **Clone the repository (if applicable) or download the notebook.**
2.  **Ensure all dependencies are installed** (see Setup and Installation).
3.  **Place your `kaggle.json` file** in the root directory or be ready to upload it when the notebook prompts you.
4.  **Open and run the `Bhavesh - Project 5 - DeepLearning.ipynb` notebook cells sequentially.**
    * The notebook will guide you through downloading the dataset, preprocessing, training, and evaluation.
    * The prediction section at the end allows testing the trained model on sample images.

## Results
* The model was trained for 20 epochs. [cite: 12, 14]
* **Final Training Accuracy:** ~94.65% [cite: 14]
* **Final Validation Accuracy:** ~82.84% (using the test set as validation data) [cite: 14]
* **Loss/Accuracy Plots:** The plots show that while training accuracy increases and training loss decreases, validation loss starts to increase significantly after epoch 14, and validation accuracy fluctuates/declines from its peak, indicating overfitting. [cite: 15, 17]
* **Sample Predictions:**
    * `CHNCXR_0327_1.png`: Predicted Tuberculosis (Confidence: 98.59%) [cite: 29]
    * `others (49).jpg`: Predicted Normal (Confidence: 100.00%)

## Files in this Repository
* `Bhavesh - Project 5 - DeepLearning.ipynb`: The Jupyter Notebook containing all the code.
* `Tuberculosis.h5`: The saved trained Keras model in HDF5 format. [cite: 18]
* `Tuberculosis.tflite`: The converted TensorFlow Lite model. [cite: 18]
* `README.md`: This file.

## Future Work
* **Address Overfitting:** Implement techniques like more aggressive data augmentation, L1/L2 regularization, or a different dropout strategy.
* **Dedicated Validation Set:** Split the data into three distinct sets (train, validation, test) for more robust model evaluation and hyperparameter tuning.
* **Hyperparameter Tuning:** Experiment with different learning rates, batch sizes, number of layers, and neurons per layer.
* **Explore Different Architectures:** Try pre-trained models (e.g., ResNet, VGG, EfficientNet) with transfer learning.
* **Advanced Image Augmentation:** Use more diverse augmentation techniques beyond simple rescaling.
* **Deployment:** Develop a simple web application or mobile app using the TFLite model for real-world testing.
