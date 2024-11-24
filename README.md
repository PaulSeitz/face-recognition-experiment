# Personal Face Recognition Pipeline Experiment

## Introduction

This project implements a face detection and recognition system using deep learning techniques. It uses a CNN model for face recognition and a pre-trained YuNet model for face detection. The system can be trained on custom datasets and perform real-time face detection and recognition using a webcam.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/face-recognition-project.git
   cd face-recognition-project
   ```

2. Install the required dependencies using the requirements file:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Configuration Files

1. `resources.txt`: This file specifies the classes and paths for the training data. Each line should be in the format:
   ```
   class_name: path/to/images/or/videos
   ```
   Example:
   ```
   unknown: /path/to/unknown/faces/
   Max Mustermann: /path/to/max/images/
   John Smith: /path/to/john/images/
   ```

2. `default_params.txt`: This file contains the training parameters. Modify as needed:
   ```
   epochs: 16
   batch_size: 1
   weight_decay:0.01
   learning_rate: 0.0001
   ```

### Command Line Options

Run the main script with the following options:

- `--detect`: Start real-time face detection and recognition using the (default) webcam
- `--train`: Train the CNN model
- `--finetune`: Fine-tune the CNN model
- `--resource_file`: Path to the resource file (default: './data/resources.txt')
- `--load_model`: Path to a saved model for loading
- `--training_params`: Path to the training parameter file (default: './default_params.txt')

Examples:

1. To train the model:
   ```
   python main.py --train --resource_file ./data/resources.txt
   ```

2. To run face detection and recognition:
   ```
   python main.py --detect --load_model path/to/trained_model.pth
   ```
   
3. To fine-tune the model:
   ```
   python main.py --finetune --load_model path/to/trained_model.pth --resource_file ./data/finetune_resources.txt
   ```

## Credits

- Pre-trained YuNet face detection model: [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- LFW-deepfunneled dataset: [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/)
- CNN architecture inspired by AlexNet: [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

## License

This project is open-source and available under the MIT License. Feel free to use, modify, and distribute the code for any purpose.
