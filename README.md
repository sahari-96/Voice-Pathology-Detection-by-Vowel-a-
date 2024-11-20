# Voice-Pathology-Detection-by-Vowel-a-
In this repository, we use two different features and a CNN model to classify healthy and unhealthy samples by vowel /a/ sound

# AVFAD voice disorder detection

This repository contains code for automatic voice pathology detection using the AVFAD dataset. The code extracts LPC and MFCC features from audio files and trains a CNN model to classify healthy and unhealthy samples.

## Dataset

The AVFAD dataset is used in this project. You may access the dataset via the following links:

- [https://www.intechopen.com/chapters/55960](https://www.intechopen.com/chapters/55960)
- [https://acsa.web.ua.pt/AVFAD.htm](https://acsa.web.ua.pt/AVFAD.htm)

## Feature Extraction

The code extracts two types of features from the audio files:

- **LPC (Linear Predictive Coding)**: These features represent the spectral envelope of the speech signal.
- **MFCC (Mel-Frequency Cepstral Coefficients)**: These features represent the short-term power spectrum of the speech signal.

## Model Training

A Convolutional Neural Network (CNN) model is trained to classify samples based on the extracted features. The model architecture includes convolutional layers, max pooling layers, global average pooling, dropout for regularization, and fully connected layers. The model is trained using the Adam optimizer and binary cross-entropy loss function.

## Usage

1. Download the AVFAD dataset and make train(460 samples), test(142 samples), and validation(105 samples) datasets and place it in the `drive/MyDrive` directory.
2. Mount Google Drive in Colab using the provided code.
3. Run the code to extract features and train the model.
4. Evaluate the model's performance on the test and validation sets.

## Results

The model achieves the following accuracies:

- **MFCC Features:**
    - Validation Accuracy: 0.9047
    - Test Accuracy: 0.8591
- **LPC Features:**
    - Validation Accuracy: 0.8857
    - Test Accuracy: 0.8169
## Some of the Requirements

- Python 3
- Libraries: pandas, numpy, matplotlib, tensorflow, keras, librosa, scikit-learn, glob, pathlib

## Note

The code is written for Google Colab and assumes that the dataset is stored in Google Drive. You may need to modify the code if you are using a different environment.


## Acknowledgements

The AVFAD dataset is used for this project. We thank the authors of the dataset for making it.
