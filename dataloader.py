import os
import glob
import random
import cv2 as cv
import torch
from torch.utils.data import Dataset
from face_detection import detect_faces_in_image
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils
from torchvision import transforms

random_images = []


def show_tensor_image(tensor, title="Tensor Image"):
    """
    Display a PyTorch tensor as an image.

    Args:
    tensor (torch.Tensor): A tensor of shape (C, H, W) or (H, W)
    title (str): Title for the plot
    """
    # Make sure the tensor is on CPU
    tensor = tensor.cpu().detach()

    # If tensor is (C, H, W), convert to (H, W, C)
    if len(tensor.shape) == 3:
        tensor = tensor.permute(1, 2, 0)

    # Clamp values to [0, 1] range
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to numpy array
    img = tensor.numpy()

    # Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


# recursively get all the directories (specifically needed for the lfw-deepfunneled dataset:
# https://www.kaggle.com/datasets/jessicali9530/lfw-dataset/data?select=lfw-deepfunneled / https://vis-www.cs.umass.edu/lfw/)
def get_directories(multiple_dirs: list[str]):
    directories = []
    for root_dir in multiple_dirs:
        temp_dirs = []
        for root, dirs, files in os.walk(root_dir):
            for directory in dirs:
                temp_dirs.append(os.path.join(root, directory))

        if not temp_dirs:
            temp_dirs.append(root_dir)

        directories.extend(temp_dirs)

    return directories


# show images
def show_images(images):
    cv.imshow('Image', images)
    # wait for the esc key to be pressed
    if cv.waitKey(0) & 0xFF == 27:
        return

    cv.destroyAllWindows()


# load the images from disk
def load_images_from_disk(path, curr_class, use_folders=False):
    images: (np.ndarray, str) = []

    image_paths = get_directories(path)

    # only load the first 5000 images
    if len(image_paths) > 4500:
        image_paths = image_paths[:4500]

    for sub_dir_path in image_paths:
        # scan for images
        found_images = glob.glob(sub_dir_path + '/*.jpg')
        found_images.extend(glob.glob(sub_dir_path + '/*.JPG'))
        found_images.extend(glob.glob(sub_dir_path + '/*.png'))

        for image_path in found_images:
            image = cv.imread(image_path)

            if image is None:
                continue

            # tag the image with the current class
            tag = curr_class

            if use_folders:
                tag = sub_dir_path.split('/')[-1]

            images.append((image, tag))

        # check for videos
        found_videos = glob.glob(sub_dir_path + '/*.mp4')
        found_videos.extend(glob.glob(sub_dir_path + '/*.mkv'))
        found_videos.extend(glob.glob(sub_dir_path + '/*.MOV'))
        found_videos.extend(glob.glob(sub_dir_path + '/*.GIF'))

        video_images = []
        for video_path in found_videos:
                if use_folders:
                    video_images.extend(load_videos_from_disk(video_path, sub_dir_path.split('/')[-1]))
                else:
                    video_images.extend(load_videos_from_disk(video_path, curr_class))


        images.extend(video_images)

    return images


# load videos from disk and take every x frames
def load_videos_from_disk(video_path, curr_class, frames_step=2):
    frames = []
    cap = cv.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # tag the image with the current class
        tag = curr_class

        # scale the images to 1920x1080
        frame = cv.resize(frame, (1024, 1024))

        frames.append((frame, tag))
        # skip frames
        for _ in range(frames_step):
            cap.read()

    cap.release()
    return frames


# run the images through the face detection model and return the detected faces
def detect_faces(images: list[tuple[np.ndarray, str]]):
    detected_faces: list[tuple[np.ndarray, str]] = []
    iterator = tqdm(images)
    iterator.set_description('Detecting faces')
    for image, tag in iterator:

        faces = detect_faces_in_image(image)

        if not faces:
            continue

        for face in faces[0]:
            detected_faces.append((face, tag))

    return detected_faces


def transform_faces(faces: list[tuple[np.ndarray, str]]):
    transformed_faces = []

    for face, tag in faces:
        # Original face
        transformed_faces.append((face, tag))

        # Flipped face
        flipped_face = cv.flip(face, 1)
        transformed_faces.append((flipped_face, tag))

    return transformed_faces


# get the dataset
def get_dataset(classes: dict[str, list[str]]):
    images: (np.ndarray, str) = []

    # load the images
    for item in classes.items():
        tag = item[0]
        paths: list[str] = item[1]

        curr_images = []
        # if the tag is the standard placeholder, use the subdirectory names as tags
        if tag == "[]":
            for path in paths:
                curr_images += load_images_from_disk([path], tag, use_folders=True)
        else:
            curr_images = load_images_from_disk(paths, tag)

        # add all the images to the list
        images.extend(curr_images)

    detected_faces: list[tuple[np.ndarray, str]] = detect_faces(images)

    detected_faces = transform_faces(detected_faces)

    # shuffle the dataset
    random.shuffle(detected_faces)

    return detected_faces


def get_random_image():
    return random_images


class FaceDataset(Dataset):
    def __init__(self, dataset: list[tuple[np.ndarray, str]], classes):
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.dataset = []
        for image, tag in dataset:
            if tag in self.class_to_idx:
                self.dataset.append((utils.resize_image(image), self.class_to_idx[tag]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        label_tensor = torch.zeros(len(self.classes))
        label_tensor[label] = 1
        return image, label_tensor

    def get_classes(self):
        return self.classes


def get_dataloader(classes, save_faces=False, load_faces=False, max_images_per_class=None):
    if load_faces:
        images = load_faces_from_file('./data/faces.pkl')
        with open('./data/classes.pkl', 'rb') as file:
            class_string = pickle.load(file)
    else:
        images = get_dataset(classes)
        class_string = list(set(label for _, label in images))

        # Apply data augmentation
        images = transform_faces(images)

        if save_faces:
            save_faces_to_file(images, './data/faces.pkl')
            with open('./data/classes.pkl', 'wb') as file:
                pickle.dump(class_string, file)



    return FaceDataset(images, class_string)

def save_faces_to_file(images, path):
    print("saved faces\n")
    with open(path, 'wb') as file:
        pickle.dump(images, file)


def load_faces_from_file(path):
    print("Loaded faces\n")
    with open(path, 'rb') as file:
        return pickle.load(file)