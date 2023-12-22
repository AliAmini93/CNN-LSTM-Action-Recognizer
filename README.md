# Recuurent Convolutional Action Recognizer

This project focuses on building a video classifier using the UCF101 dataset, a comprehensive collection of videos categorized into actions like cricket shots, punching, biking, etc.

## Conceptual Foundations

**Understanding the Dataset**: The UCF101 dataset is widely recognized for building action recognizers, a key application of video classification. It comprises videos, each an ordered sequence of frames. These frames carry spatial information, while their sequencing imparts temporal information.

**Architectural Approach**: To effectively model both spatial and temporal aspects, a hybrid architecture combining Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) is employed. This architecture, known as CNN-RNN or CNN-LSTM, integrates the spatial processing strengths of CNNs with the temporal processing capabilities of RNNs, specifically using GRU (Gated Recurrent Unit) and LSTM (Long Short-Term Memory) layers.

## Technical Prerequisites

* TensorFlow 2.5 or higher is required.
* A subsampled version of the UCF101 dataset is utilized in this project. The process of subsampling is detailed in [this notebook](https://github.com/AliAmini93/CNN-LSTM-Action-Recognizer/blob/main/UCF101_Data_Preparation_Top5.ipynb), which provides insights into how the dataset was prepared for this specific application.

# Models and Configuration Parameters

This project implements two distinct models for video classification. The first model utilizes a pre-trained InceptionV3 (trained on ImageNet) as the CNN-based spatial feature extractor combined with a GRU layer for temporal feature extraction. The second model employs EfficientNetB7 as the CNN component and an LSTM layer for processing temporal features. These models are employed to capture and learn from both spatial and temporal aspects of the video data.

## Configuration Parameters

Specific parameters are set for image processing, model training, and sequence processing in video frames. These parameters ensure the model processes and learn from the data.

### Image Processing and Model Training Parameters

- `IMAGE_DIMENSION`: 600, 224 (EfficientNetB7, InceptionV3)
  The target size for resizing images. This dimension is used to standardize the size of the input images for consistent processing. Two sizes are used for the different models.

- `BATCH_SIZE`: 64  
  Defines the number of samples that will be propagated through the network. A batch size of 64 is used for training the model.

- `TRAINING_EPOCHS`: 60  
  The number of complete passes through the training dataset. The models will be trained for 60 epochs, with early stopping implemented to prevent overfitting if the validation loss does not improve after 15 epochs.

### Sequence Processing Parameters

- `SEQUENCE_LENGTH`: 20  
  The maximum length of the sequence of frames in a video. This parameter sets the number of frames to be considered for each video, ensuring uniformity in temporal dimension across all videos.

- `FEATURE_VECTOR_SIZE`: 2560, 2048 (EfficientNetB7, InceptionV3)
  The number of features to be extracted from each frame. These feature vector sizes are crucial for capturing the necessary information from each frame for successful classification. Different sizes are utilized for the different models.

These configuration parameters play a pivotal role in the models' ability to learn from the video data and accurately classify actions, optimizing performance while balancing computational efficiency.




