import os

import face_detection
import argparse
import train
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description='Real time face detection and recognition pipeline')
    parser.add_argument('--detect', action='store_true', help='Detect faces in a video feed')
    parser.add_argument('--train', action='store_true', help='Train the CNN model')
    parser.add_argument('--finetune', action='store_true', help='Fine tune the CNN model on provided data')
    parser.add_argument('--resource_file', type=str, default='./data/resources.txt',
                        help='Path to the resource file that includes the classes and paths to corresponding '
                             'resources (image / video) [recursively scrapes all subdirectories]')
    parser.add_argument('--load_model', type=str, default='', help='Path to a saved model')
    parser.add_argument('--training_params', type=str, default='./default_params.txt',
                        help='Path to a training parameter file')
    return parser.parse_args()


def extract_training_params(path):
    with open(path, 'r') as file:
        prms = file.readlines()
    return {param.split(":")[0]: param.split(":")[1].strip() for param in prms}


def traverse_directories(path):
    # get all the lowest level directories in the given path
    directories = []
    for root, dirs, files in os.walk(path):
        if not dirs:
            directories.append(root)
    return directories


def extract_classes_and_paths(path) -> dict[str, list[str]]:
    # open the file and split it into [class, [paths]] (split by : and the paths split by ;)
    with open(path, 'r') as file:
        lines = file.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].replace("\n", "")
        lines[i].strip(" ")

    output = {cls.split(":")[0]: cls.split(":")[1].split(";") for cls in lines}
    output = {key: [path.strip() for path in paths] for key, paths in output.items()}

    new_classes = {}
    # check if one of the classes is a placeholder * and replace it with the folder names in the given directory
    for key, paths in output.items():
        if key == "*":
            for path in paths:
                directories = traverse_directories(path)
                for directory in directories:
                    new_classes[directory.split("/")[-1]] = [directory]

    if "*" in output:
        output.pop("*")

    # insert the new classes into the output
    for key, paths in new_classes.items():
        output[key] = paths

    # strip the paths from leading and trailing whitespaces / newlines
    return output


if __name__ == '__main__':

    args = parse_args()

    if args.detect:
        if args.load_model == '':
            print("No model provided, please provide a model to detect faces")
            exit(1)
        face_detection.start_recording(args.load_model, DEVICE)
    elif args.train:
        classes = extract_classes_and_paths(args.resource_file)
        params = extract_training_params(args.training_params)
        train.train(classes, params, args.load_model)
    elif args.finetune:
        print("currently not implemented")
    else:
        print("Something went wrong, a modus is required")