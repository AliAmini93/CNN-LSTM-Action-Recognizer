# Recuurent Convolutional Action Recognizer

This project focuses on building a video classifier using the UCF101 dataset, a comprehensive collection of videos categorized into various actions like cricket shots, punching, biking, etc.

## Conceptual Foundations

**Understanding the Dataset**: The UCF101 dataset is widely recognized for building action recognizers, a key application of video classification. It comprises videos, each an ordered sequence of frames. These frames carry spatial information, while their sequencing imparts temporal information.

**Architectural Approach**: To effectively model both spatial and temporal aspects, a hybrid architecture combining Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) is employed. This architecture, known as CNN-RNN or CNN-LSTM, integrates the spatial processing strengths of CNNs with the temporal processing capabilities of RNNs, specifically using GRU (Gated Recurrent Unit) and LSTM (Long Short-Term Memory) layers.

## Technical Prerequisites

* TensorFlow 2.5 or higher is required.
* A subsampled version of the UCF101 dataset is utilized in this project. The process of subsampling is detailed in [this notebook](https://github.com/AliAmini93/CNN-LSTM-Action-Recognizer/blob/main/UCF101_Data_Preparation_Top5.ipynb), which provides insights into how the dataset was prepared for this specific application.

