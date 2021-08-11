# Introduction
I have created an application that automatically calculates darts scores by applying object detection. As you can see in the gif below, it detects the arrow stuck in the dartboard and displays the score. 

![demo](/img/demo.gif)

This application detects the Bull (the center of the dartboard) and the arrow by transfer learning from SSD-Mobilenet. In general, SSD-Mobilenet can detect objects such as dartboards and arrows, but it is difficult to determine the score. For example, since there are 61 different patterns of dart scores, which are combinations of numbers 1-20 and multiples (Single, Double, Triple) + Bull, we have to make sure that a part of the dartboard is detected accordingly. Therefore, in this project, I created the original dataset and added a neural network to estimate the scores, thus making Darts Score Detection a reality.


# Table of contents
 - [Requirements](#requirements)
 - [Advance preparation](#advance-preparation)
 - [SetUp](#setup)
 - [File Details](#file-details)
 - [Running the application](#running-the-application)
 - [How it works](#how-it-works)
 - [Data collection](#data-collection)
 - [Training for area discrimination](#training-for-area-discrimination)
 - [Rewriting annotations](#rewriting-annotations)
 - [Tranfer Learning with SSD-Mobilenet](#tranfer-learning-with-ssd-mobilenet)
 - [Future Direction](#future-direction)

# Requirements
 - Hardware
  	+ [Jetson Nano](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/education-projects/)
  	+ [Logitech HD Pro Webcam C920](https://www.amazon.com/Logitech-Widescreen-Recording-Certified-Refurbished/dp/B010BJJAVY/ref=sr_1_3?dchild=1&keywords=C920n&qid=1621944348&sr=8-3)
  	+ Memory card(64GB)

 - Software
  	+ Python==3.6.9
  	+ torch==1.6.0
  	+ onnxruntime==1.8.0

# Advance preparation
Before starting this project, please follow the information below to set up.

 - [Install JetPack SDK](https://developer.nvidia.com/embedded/jetpack)
 - [Detailed setup videos](https://developer.nvidia.com/embedded/learn/jetson-ai-certification-programs#collapseTwo)
 - [Running the Docker Container](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-docker.md)


# SetUp
Clone this project from the GitHub repository.

```
  git@github.com:kawasaki-kento/darts-score-detection.git
```

Connect the camera (C920) to the Jetson Nano and verify that the camera recognizes it with the following code.

```
  camera-capture /dev/video0
``` 

Once you have confirmed that the camera is connected, adjust the position of the dartboard and the camera. The camera should be positioned directly in front of the dartboard, and the distance between the camera and the dartboard should be about 60cm to 80cm. If necessary, use a tripod.

# File Details
```
darts-score-detection
├── models
│   ├── dnn 
│   │   ├── dnn_model.pth  : Neural Network Model for area discrimination
│   │   └── dnn_model.onnx : Neural network model for area discrimination (ONNX format)
│   └── ssd
│       ├── score_labels.txt    : Label before annotation change
│       ├── labels.txt          : Label after annotation change
│       └── ssd-mobilenet.onnx  : Trained SSD-Mobilenet to detect Bull and Arrow
├── darts_score_detection.py    : Running Darts Score Detection
├── change_annotations.py       : Change the annotations
├── feature_creation.py         : Create features from annotated data
├── score_detection_training.py : Run training for area discrimination
└── README.md
```

# Running the application
Mount the "darts_score_detection" directory in a container and run darts_score_detection.py as shown below. For details on each argument, please refer to [here](https://github.com/dusty-nv/jetson-inference/blob/master/docs/detectnet-console-2.md).

```
  cd jetson-inference
  docker/run.sh --volume ~/darts_score_detection:/darts_score_detection
  python3 darts_score_detection.py
   --model=./models/ssd/ssd-mobilenet.onnx
   --labels=./models/ssd/labels.txt
   --input-blob=input_0
   --output-cvg=scores
   --output-bbox=boxes
   --threshold=0.3 /dev/video0
```

When you run it, the camera capture screen will open and the Bull (center of the dartboard) should be detected.

![detecting bull](/img/Bull.PNG)

After confirming that the Bull (the center of the dartboard) has been successfully detected, the game will start. Try to shoot arrows at the dartboard. If all goes well, you should see the scores as follows.

![detecting arrow](/img/Arrow.PNG)

# How it works
This application uses SSD-Mobilenet, but it only detects the Bull (center of the dartboard) and the arrow. SSD-Mobilenet alone is not enough to estimate the score. To determine the score, we use the information of the position and angle of the arrow from the center point of the dartboard. The flow of score calculation is as follows.


![how_it_works](/img/how_it_works.png)

 - When the application is launched, SSD-Mobilenet detects the Bull.
 - The user throws an arrow at the dartboard.
 - When the arrow sticks the dartboard, SSD-Mobilenet detects the arrow.
 - Estimates the score (1-20) based on the relative angle of the arrow to the Bull (center of the dartboard).
 - Estimates multiples (Single, Double, Triple) from four features: the distance of the arrow relative to the Bull (center of the dartboard), the angle, and the width and height of the arrow's bounding box.
 - We use a neural network to estimate the multiples.
 - If the relative distance of the arrow to the Bull (the center of the dartboard) is extremely close, it is estimated as Bull.




# Data collection
To collect data, I stabed arrows at the dartboard and annotated the Bull (the center of the dartboard) and the arrows. The annotation process was done with the camera set up so that the dartboard was directly in front of it, as shown in the gif below.

![annotations](/img/annotations.gif)

The arrows were annotated with the score of the position where they were stuck as labels. Thus, the number of labels is 60 (20 kinds of scores x 3 kinds). I have collected more than 1000 such annotation data in the following directory.

```
/jetson-inference/python/training/detection/ssd/data/darts_score_detection/Annotations
```
This annotation data will be used to Training for area discrimination at first, but later it will be rewritten and used to Tranfer Learning in SSD-Mobilenet.


# Training for area discrimination
There are single, double, and triple areas on the dartboard, but the relative distance from the Bull (the center point of the dartboard) alone is not accurate enough to identify them. So, using a neural network, I implemented a model to determine the area where the arrow was stuck. The following code will create four features from the data contained in the "Annotaions" folder created during data collection: Bounding Box Width, Bounding Box Height, Distance from Bull (center point of the dartboard), and Angle from Bull (center point of the dartboard).

```
python feature_creation.py
 --annotations-dir=/jetson-inference/python/training/detection/ssd/data/darts_score_detection/Annotations
 --output-file=/jetson-inference/python/training/detection/ssd/data/darts_score_detection/features.tsv
```

Using the created features, train a deep learning model that predicts the area where the arrow is stuck. If you set onnx-option to True, it will also output ONNX format files.
 
```
python score_detection_training.py
 --input-data=/jetson-inference/python/training/detection/ssd/data/darts_score_detection/features.tsv
 --output-dir=./models/dnn/
 --split-percent=0.6
 --train-epochs=500
 --onnx-option=True
```
For inference, you can use onnxruntime and write the following code to perform model-based inference.

```Python
ort_session = onnxruntime.InferenceSession("./models/dnn/dnn_model.onnx")
ort_inputs = {ort_session.get_inputs()[0].name: [[distance, rad, box_width, box_height]]}
multiple = np.argmax(ort_session.run(None, ort_inputs)[0])
```
The trained dnn_model.onnx will be used in darts_score_detection.py to predict the area where the arrow will stick.

# Rewriting annotations
If we use the annotations created during data collection, we will end up with a model with low accuracy, so we will change the annotation data. There are 60 different labels for arrows, all of which have been replaced by the label "Arrow". 

![rewriting_annotations](/img/rewriting_annotations.PNG)

This way, SSD-Mobilenet is only in charge of detecting the arrows. To change the annotation data, change the Annotaions and ImageSets of the data created in the data collection as shown below.

```
python change_annotations.py
 --labels-txt=/jetson-inference/python/training/detection/ssd/data/darts_score_detection/labels.txt
 --new-label=Arrow
 --annotations-dir=/jetson-inference/python/training/detection/ssd/data/darts_score_detection/Annotations
 --new-annotations-dir=/jetson-inference/python/training/detection/ssd/data/darts_score_detection/NewAnnotations
```

# Tranfer Learning with SSD-Mobilenet
I ran transfer learning with SSD-Mobilenet using data with rewritten annotations. I specified the directory containing the data collected in the data collection (labels rewritten) and ran train_ssd.py as shown below. Since the training epoch was 200 and long study, it is recommended to do it on a desktop PC with GPU. The detailed transfer learning method using SSD-Mobilenet is described below.

[Re-training SSD-Mobilenet](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md)

```
   cd jetson-inference
   docker/run.sh
   cd python/training/detection/ssd
   python3 train_ssd.py
    --dataset-type=voc
    --data=data/darts_score_detection
    --model-dir=models/darts_score_detection
    --batch-size=16
    --workers=1
    --epochs=200
```

When the training is finished, a trained model will be created, and now we will convert it to an ONNX format file using onnx_export.py.

```
   cd /jeston-inference/python/training/detection/ssd
   python3 onnx_export.py --model-dir=models/darts_score_detection
```

# Future Direction
The current model has been learned from thousands of data, so the prediction accuracy is still low. In the future, I plan to improve the prediction accuracy by processing more informative data, such as predicting the score from the image data of the bounding box using CNN, in addition to the relative distance and angle of the arrow to the Bull (the center of the dartboard).

# References
 - [dusty-nv/jetson-inference](https://github.com/dusty-nv/jetson-inference)