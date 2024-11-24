import os
import glob
import random
import cv2 as cv
import torch
from torch.utils.data import Dataset
from face_detection import detect_faces_in_image
from collections import OrderedDict
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils
from scipy.ndimage import rotate

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

        # scale the images to 1024 1024
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


def transform_faces(faces: list[tuple[np.ndarray, str]], augmentation_factor=None, output_dir="./data/augmented_faces"):
    """
    Apply face-specific augmentations with dynamic augmentation
    returns paths to saved augmented images instead of keeping them in memory

    :param faces: List of (image, label)
    :param augmentation_factor: Number of augmentations to apply per face
    :param output_dir: Directory to save augmented faces
    :return: List of (path, label) tuples
    """
    class_counts = analyze_raw_labels(faces)
    target_size = max(class_counts.values()) if class_counts else 1

    transformed_faces = []

    # Group faces by class
    class_faces = {}
    for face, label in faces:
        if label not in class_faces:
            class_faces[label] = []
        class_faces[label].append(face)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Clean the output directory
    for file in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, file))

    for label, class_face_list in class_faces.items():
        current_size = len(class_face_list)
        if augmentation_factor is None:
            aug_factor = min(40, max(2, int(target_size / current_size)))
        else:
            aug_factor = augmentation_factor

        iterator = tqdm(class_face_list)
        iterator.set_description(f'Augmenting faces with factor {aug_factor}')

        for i, face in enumerate(iterator):
            # Convert and save original face
            processed_face = utils.resize_image(face).permute(1, 2, 0).numpy()
            processed_face = (processed_face * 255).astype(np.uint8)
            orig_path = os.path.join(output_dir, f"orig_{label}_{i}.jpg")
            cv.imwrite(orig_path, processed_face)
            transformed_faces.append((orig_path, label))

            for j in range(aug_factor):
                augmented = face.copy()

                # Focus on photometric augmentations only
                # 1. Brightness/Contrast changes
                if random.random() < 0.5:
                    brightness_factor = random.uniform(0.8, 1.2)
                    augmented = cv.convertScaleAbs(augmented, alpha=brightness_factor)
                    contrast_factor = random.uniform(0.8, 1.2)
                    mean = np.mean(augmented)
                    augmented = (augmented - mean) * contrast_factor + mean
                    augmented = np.clip(augmented, 0, 255).astype(np.uint8)

                # 2. Color jittering
                if random.random() < 0.3:
                    # Randomly adjust individual color channels
                    for channel in range(3):
                        factor = random.uniform(0.9, 1.1)
                        augmented[:, :, channel] = np.clip(augmented[:, :, channel] * factor, 0, 255)

                # 3. Mild noise addition
                if random.random() < 0.3:
                    noise = np.random.normal(0, 3, augmented.shape).astype(np.uint8)
                    augmented = cv.add(augmented, noise)

                # 4. Horizontal flip (only geometric augmentation kept as it doesn't affect alignment)
                if random.random() < 0.5:
                    augmented = cv.flip(augmented, 1)

                # Save preprocessed augmented face
                processed_augmented = utils.resize_image(augmented).permute(1, 2, 0).numpy()
                processed_augmented = (processed_augmented * 255).astype(np.uint8)
                aug_path = os.path.join(output_dir, f"aug_{label}_{i}_{j}.jpg")
                cv.imwrite(aug_path, processed_augmented)
                transformed_faces.append((aug_path, label))

    return transformed_faces


def analyze_raw_labels(faces: list[tuple[np.ndarray, str]]) -> dict:
    """
    Analyze the distribution of raw string labels in the dataset
    :param faces: List of (image, label) tuples

    :return dict: Dictionary mapping class names to counts
    """
    class_counts = {}
    for _, label in faces:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    return class_counts


class FaceDataset(Dataset):
    """
    Hybrid dataset class that combines in-memory caching with on-disk storage
    """

    def __init__(self, dataset: list[tuple[str, str]], classes, cache_size=2000):
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.cache_size = cache_size
        self.cache = OrderedDict()  # Using OrderedDict for LRU cache

        # Store image paths and labels
        self.dataset = []
        for image_path, tag in dataset:
            if tag in self.class_to_idx:
                self.dataset.append((image_path, self.class_to_idx[tag]))

        random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def _load_image(self, image_path):
        """Load and preprocess image from disk"""
        try:
            image = cv.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                return None

            return utils.resize_image(image)

        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None

    def __getitem__(self, idx):
        image_path, label = self.dataset[idx]

        # Try to get from cache first
        cache_key = image_path
        if cache_key in self.cache:
            face_tensor = self.cache[cache_key]
        else:
            # Load from disk
            face_tensor = self._load_image(image_path)
            if face_tensor is None:
                # Load next valid image instead of returning zeros
                next_idx = (idx + 1) % len(self)
                return self.__getitem__(next_idx)

            # Update cache using path as key
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)  # Remove oldest item
            self.cache[cache_key] = face_tensor

        # Create one-hot encoded label
        label_tensor = torch.zeros(len(self.classes))
        label_tensor[label] = 1

        return face_tensor, label_tensor

    def get_classes(self):
        return self.classes


def get_datasets(classes: dict[str, list[str]], save_faces=True, data_path="./data/training_data/"):
    """
    Creates the training and evaluation datasets from the provided classes and paths
    If save_faces is True, will save the datasets to disk for future use, otherwise will load them from disk
    """
    images: list[tuple[np.ndarray, str]] = []
    augmented_dir = os.path.join(data_path, "augmented_faces")

    if save_faces:
        # Load the images
        for tag, paths in classes.items():
            curr_images = []
            if tag == "[]":
                for path in paths:
                    curr_images += load_images_from_disk([path], tag, use_folders=True)
            else:
                curr_images = load_images_from_disk(paths, tag)

            images.extend(curr_images)


        # filter out images with less than 10 faces
        class_counts = analyze_raw_labels(images)
        images = [(image, label) for image, label in images if class_counts[label] >= 10]

        # Shuffle dataset
        random.shuffle(images)

        # Get unique classes
        class_string = sorted(list(set(label for _, label in images)))

        # Split for training and eval
        split_idx = int(len(images) * 0.8)
        train_images = images[:split_idx]
        eval_images = images[split_idx:]

        # Process train and eval sets with permanent storage location
        train_faces = transform_faces(detect_faces(train_images), output_dir=os.path.join(augmented_dir, "train"))
        eval_faces = transform_faces(detect_faces(eval_images), output_dir=os.path.join(augmented_dir, "eval"))

        # Create datasets with hybrid loading
        train_dataset = FaceDataset(train_faces, class_string, cache_size=5000)
        eval_dataset = FaceDataset(eval_faces, class_string, cache_size=2000)

        # Save datasets metadata to disk
        torch.save({
            'dataset_paths': train_faces,
            'classes': class_string,
            'cache_size': train_dataset.cache_size
        }, f"{data_path}train_full.pt")

        torch.save({
            'dataset_paths': eval_faces,
            'classes': class_string,
            'cache_size': eval_dataset.cache_size
        }, f"{data_path}eval_full.pt")

        class_counts = {}
        for _, label in train_dataset:
            label_idx = torch.argmax(label).item()
            class_counts[label_idx] = class_counts.get(label_idx, 0) + 1
        print("Class distribution:", class_counts)

        return train_dataset, eval_dataset, class_string

    else:
        # Load the saved datasets
        try:
            train_data = torch.load(f"{data_path}train_full.pt")
            eval_data = torch.load(f"{data_path}eval_full.pt")

            # Create new dataset instances
            train_dataset = FaceDataset(
                train_data['dataset_paths'],
                train_data['classes'],
                train_data['cache_size']
            )

            eval_dataset = FaceDataset(
                eval_data['dataset_paths'],
                eval_data['classes'],
                eval_data['cache_size']
            )

            # Verify files exist
            def verify_dataset(dataset):
                for image_path, _ in dataset.dataset:
                    if not os.path.exists(image_path):
                        return False
                return True

            if not verify_dataset(train_dataset) or not verify_dataset(eval_dataset):
                print("Some augmented face files are missing, regenerating datasets...")
                return get_datasets(classes, save_faces=True, data_path=data_path)

            return train_dataset, eval_dataset, train_data['classes']

        except Exception as e:
            print(f"Error loading datasets: {e}")
            print("Regenerating datasets...")
            return get_datasets(classes, save_faces=True, data_path=data_path)


def get_random_image():
    """
    Get random images from the dataset
    :return: a list of random images
    """
    return random_images

def save_faces_to_file(images, path):
    """
    Save detected faces to a file
    :param images: list of detected faces
    :param path: path to save the faces to
    """
    with open(path, 'wb') as file:
        pickle.dump(images, file)


def load_faces_from_file(path):
    """
    Load detected faces from a file
    :param path: path to load the faces from
    :return the detected faces
    """
    with open(path, 'rb') as file:
        return pickle.load(file)