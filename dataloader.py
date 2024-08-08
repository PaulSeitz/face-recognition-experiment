import os
import glob
import random
import cv2 as cv
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from face_detection import detect_faces_in_image
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

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
def load_images_from_disk(path, curr_class):
    images: (np.ndarray, str) = []

    image_paths = get_directories(path)

    for sub_dir_path in image_paths:
        # scan for images
        found_images = glob.glob(sub_dir_path + '/*.jpg')
        found_images.extend(glob.glob(sub_dir_path + '/*.JPG'))
        found_images.extend(glob.glob(sub_dir_path + '/*.png'))

        for image_path in found_images:
            image = cv.imread(image_path)

            if image is None:
                continue

            # only use the first 2000 images
            if len(images) > 1000:
                break

            # tag the image with the current class
            tag = curr_class

            images.append((image, tag))

        # check for videos
        found_videos = glob.glob(sub_dir_path + '/*.mp4')
        found_videos.extend(glob.glob(sub_dir_path + '/*.mkv'))
        found_videos.extend(glob.glob(sub_dir_path + '/*.MOV'))
        found_videos.extend(glob.glob(sub_dir_path + '/*.GIF'))

        video_images = []
        for video_path in found_videos:
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

        faces: list[np.ndarray] = detect_faces_in_image(image)

        if not faces:
            continue

        for face in faces:
            detected_faces.append((face, tag))

    return detected_faces


# transform the retrieved faces to increase the amount of training data
def transform_faces(faces: list[tuple[np.ndarray, str]]):
    transformed_faces = []

    for face, tag in faces:

        transformed_faces.append((face, tag))

        if not isinstance(face, np.ndarray):
            continue

        # Ensure face is a 3D array (height, width, channels)
        if len(face.shape) != 3:
            continue

        # flip the image
        flipped_face = cv.flip(face, 1)
        transformed_faces.append((flipped_face, tag))

        # rotate the face by 45 degrees
        rows, cols, _ = face.shape
        rotation_matrix = cv.getRotationMatrix2D((cols/2, rows/2), 45, 1)
        rotated_face = cv.warpAffine(face, rotation_matrix, (cols, rows))
        transformed_faces.append((rotated_face, tag))

        # rotate the face by -45 degrees
        rotation_matrix = cv.getRotationMatrix2D((cols/2, rows/2), -45, 1)
        rotated_face = cv.warpAffine(face, rotation_matrix, (cols, rows))
        transformed_faces.append((rotated_face, tag))

    return transformed_faces


# get the dataset
def get_dataset(classes: dict[str, list[str]]):
    images: (np.ndarray, str) = []

    # load the images
    for item in classes.items():
        tag = item[0]
        paths: list[str] = item[1]
        curr_images = load_images_from_disk(paths, tag)

        # add all the images to the list
        images.extend(curr_images)

    # resize all the images to 400x400
    for idx, (image, tag) in enumerate(images):
        images[idx] = (cv.resize(image, (1024, 1024)), tag)

    number_of_images = len(images)

    detected_faces: list[tuple[np.ndarray, str]] = detect_faces(images)

    detected_faces = transform_faces(
        detected_faces
    )

    number_detected_faces = len(detected_faces)

    print(f'The number of images is {number_of_images} and the number of detected faces is {number_detected_faces}')

    # shuffle the dataset
    random.shuffle(detected_faces)

    return detected_faces


def resize_image(img, size=(128, 128)):

    # convert to black and white (i.e. delete the 3rd dimension)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Define the transformation to apply to the face images
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply the transformation
    img_tensor = transform(img)

    return img_tensor


def get_random_image():
    return random_images


class FaceDataset(Dataset):
    def __init__(self, dataset: list[tuple[np.ndarray, str]]):
        self.dataset = dataset

    def __len__(self):
        return int(len(self.dataset))

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = resize_image(data[0])
        label = data[1]
        return img, label


def get_dataloader(classes, save_faces=False, load_faces=False):
    if load_faces:
        images = load_faces_from_file('./data/faces.pkl')
    else:
        images = get_dataset(classes)

        classes_list = list(classes.keys())

        # Transform tags to numbers
        for idx, (img, label) in enumerate(images):
            class_idx = classes_list.index(label) if label in classes_list else 0
            images[idx] = (img, class_idx)

        if save_faces:
            save_faces_to_file(images, './data/faces.pkl')

    # print the number of each class
    for i in range(len(classes)):
        print(f"Class {i} has {len([label for _, label in images if label == i])} images")

    return FaceDataset(images)


def save_faces_to_file(images, path):
    print("saved faces\n")
    with open(path, 'wb') as file:
        pickle.dump(images, file)


def load_faces_from_file(path):
    print("Loaded faces\n")
    with open(path, 'rb') as file:
        return pickle.load(file)
