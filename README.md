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

## Video Processing Methodology

One of the primary challenges in training video classifiers is devising a method to efficiently feed videos into a neural network. Various strategies exist, as discussed in [this blog post](https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5). Given that a video is essentially an ordered sequence of frames, a straightforward approach might be to extract these frames and form a 3D tensor. However, due to varying frame counts across different videos, this method can be problematic for batching unless padding is used.

### Adopted Approach

In this project, we adopt a method similar to that used in text sequence problems. Our approach involves:

1. Capturing frames from each video.
2. Extracting frames until a maximum frame count is reached.
3. Padding videos with fewer frames than the maximum with zeros.

This method is particularly suitable for the UCF101 dataset, where there isn't significant variation in objects and actions across frames. However, it's important to note that this approach might not generalize well to other video classification tasks. We utilize [OpenCV's `VideoCapture()` method](https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html) for reading frames from videos.

### Implemented Functions

The following functions are adapted from a TensorFlow tutorial on action recognition:

#### `center_crop_frame(input_frame)`

This function crops the center square of a given frame.

- `input_frame`: The frame to be cropped.
- Returns: Cropped frame.

The function calculates the center of the frame and crops it to form a square, ensuring uniform frame dimensions.

#### `process_video(video_path, max_frame_count, target_size)`

Processes and extracts frames from a video.

- `video_path`: Path to the video file.
- `max_frame_count`: Maximum number of frames to extract.
- `target_size`: The dimensions to which each frame is resized.

The function reads the video, applies center cropping to each frame, resizes them, and reorders color channels. It then returns the processed frames as an array, adhering to the specified maximum frame count.

## Feature Extraction Using Pre-Trained Models

To extract features from the frames of each video, leveraging a pre-trained network is a highly effective approach. The [`Keras Applications`](https://keras.io/api/applications/) module offers several state-of-the-art models pre-trained on the ImageNet-1k dataset. For this project, we specifically utilize the [InceptionV3](https://arxiv.org/abs/1512.00567), and [EfficientNetB7](https://arxiv.org/pdf/1905.11946.pdf) known for their efficiency and accuracy in image classification tasks.

### InceptionV3 & EfficientNetB7 for Feature Extraction

The InceptionV3 and EfficientNetB7 models, pre-trained on ImageNet, are utilized to extract features from the video frames.

#### Function: `create_feature_extraction_model()`

This function builds a feature extraction model using the InceptionV3 and EfficientNetB7 architectures.

- Returns: A Keras model specifically designed for feature extraction.

This setup results in a robust feature extraction model that can be applied to each frame of the videos.






