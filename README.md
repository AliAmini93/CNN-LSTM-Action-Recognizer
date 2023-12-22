# Recurrent Convolutional Action Recognizer

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

An important step in preparing the dataset for training involves converting the string labels of each video into numeric indices. This conversion enables the model to process and learn from these labels effectively.

### Creating a Label Processor

We implement a label processor using Keras' `StringLookup` layer, which converts string labels into their corresponding numeric indices.

#### Functionality of `StringLookup`

- `num_oov_indices`: Set to 0, indicating the number of out-of-vocabulary indices.
- `vocabulary`: The unique tags obtained from the training data. This creates a consistent mapping from string labels to numeric indices.

## Preparing Videos: Feature Extraction and Mask Creation

To prepare the videos for the neural network, we need to extract features from each frame and create masks to handle varying video lengths. This process is essential for transforming raw video data into a format suitable for model training.

### Function: `process_videos_and_extract_features(dataframe, directory_path)`

This function prepares all videos in a given dataframe by extracting features and creating masks.

- `dataframe`: Contains information about the videos, like names and labels.
- `directory_path`: The root directory where the videos are stored.

The function processes each video, extracts frame features using the previously defined feature extraction model, and creates masks to account for videos with frame counts less than the maximum sequence length.

1. **Input Layers**: There are two input layers, one for the frame features and another for the sequence mask.

2. **GRU and LSTM Layers**: The GRU and LSTM layers are designed to process the sequence of frame features, taking into account the sequence mask. This helps the model focus on the relevant parts of the video.

3. **Output Layer**: The final layer is a dense layer with a softmax activation function, corresponding to the number of unique tags (classes) in the dataset.

## Model Training and Evaluation

In this phase, we focus on training the RNN sequence model and evaluating its performance on the test dataset. The process involves using callbacks for efficient training and the preservation of the best-performing model weights.

### Conducting the Experiment

The `conduct_experiment()` function encapsulates the entire process of training and evaluating the model.

## Predicting Video Sequences and Visualization

The final aspect of the project involves predicting actions in videos and visualizing the results. This process includes preparing the video frames for prediction, performing the sequence prediction, and converting the video frames to a GIF for an easy-to-understand visual representation.

### Preparing Video for Prediction

#### Function: `prepare_video_for_prediction(video_frames)`

Prepares a single video's frames for prediction by the sequence model.

- `video_frames`: Frames of the video to be processed.
- Returns: Processed frame features and frame mask.

The function processes each frame, extracts the features, and creates a mask to handle videos with fewer frames than the maximum sequence length.

### Performing Sequence Prediction

#### Function: `predict_video_sequence(video_path)`

Performs sequence prediction on a given video.

- `video_path`: Path to the video file.
- Returns: Frames of the video.

The function predicts the probability of each class for the given video and prints the predictions.

### Visualization Utility: Converting Frames to GIF

#### Function: `frames_to_gif(video_frames)`

Converts a sequence of video frames into a GIF.

- `video_frames`: Frames of the video.
- Returns: IPython Image display object of the created GIF.

This utility function is useful for visualizing the video frames in a more engaging and understandable format.
## Next Steps and Future Work

While the current project establishes a solid foundation for video classification, there are several avenues for future enhancements and experiments to further improve performance and adaptability.

### Fine-Tuning Pre-Trained Networks

- **Fine-Tuning**: Experiment with fine-tuning the pre-trained CNN-based networks (like InceptionV3 or EfficientNetB7) used for feature extraction. Adjusting these networks specifically for your dataset can potentially improve results.

### Exploring Model Variants

- **Speed-Accuracy Trade-offs**: Investigate other models within `keras.applications` to balance speed and accuracy. Each model offers different benefits and compromises.

- **Sequence Length Variations**: Experiment with different values for `MAX_SEQ_LENGTH`. Observe how altering the maximum sequence length affects performance.

- **Training on More Classes**: Expand the number of classes in the training dataset to challenge the model's ability to generalize and handle more diverse data.

### Advanced Techniques and Models

- **Pre-Trained Action Recognition Models**: Utilize [pre-trained action recognition models](https://arxiv.org/abs/1705.07750) like those from DeepMind, as detailed in [this TensorFlow tutorial](https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub).

- **Rolling-Averaging Technique**: Implement rolling-averaging with standard image classification models for video classification. [This tutorial](https://www.pyimagesearch.com/2019/07/15/video-classification-with-keras-and-deep-learning/) provides insights into using this technique.

- **Self-Attention for Frame Importance**: In scenarios with significant variations between frames, incorporating a [self-attention layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Attention) in the sequence model can help focus on the most relevant frames for classification.

- **Transformers for Video Processing**: Explore the implementation of Transformers-based models for processing videos, as explained in [this book chapter](https://livebook.manning.com/book/deep-learning-with-python-second-edition/chapter-11). Transformers can offer significant advantages in understanding the complex temporal dynamics in videos.

### Data Augmentation

- **Augmentation Techniques**: Implement data augmentation techniques to increase the diversity of the training dataset, which can lead to better generalization and robustness of the model.

## Acknowledgements

A heartfelt thank you to [Sayak Paul](https://twitter.com/RisingSayak) for his invaluable contribution and insight.
