import os
import cv2 as cv
from face_detection import detect_faces_in_image

IMAGE_PATHS = []
VIDEO_PATHS = []


# show images
def show_images(images):
    for image in images:
        cv.imshow('Image', image)
        # wait for the esc key to be pressed
        if cv.waitKey(0) & 0xFF == 27:
            break

        cv.destroyAllWindows()


# load the images from disk
def load_images_from_disk(image_paths):
    images = []
    for image_path in image_paths:
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # tag the image with its folder name
        tag = os.path.basename(os.path.dirname(image_path))
        images.append({image, tag})

    return images


# load videos from disk and take every x frames
def load_videos_from_disk(video_paths, frames_step=15):
    frames = []
    for video_path in video_paths:
        cap = cv.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # get the tag
            tag = os.path.basename(os.path.dirname(video_path))
            frames.append({frame, tag})
            # skip frames
            for _ in range(frames_step):
                cap.read()

    return frames


# run the images through the face detection model and return the detected faces
def detect_faces(images):
    detected_faces = []
    for image, tag in images:
        faces = detect_faces_in_image(image)
        detected_faces[tag].append(faces)

    return detected_faces


# transform the retrieved faces to increase the amount of training data
def transform_faces(faces):
    transformed_faces = []
    for face in faces:
        # flip the face horizontally
        flipped_face = cv.flip(face, 1)
        transformed_faces.append(flipped_face)
        # rotate the face by 45 degrees
        rows, cols, _ = face.shape
        rotation_matrix = cv.getRotationMatrix2D((cols/2, rows/2), 45, 1)
        rotated_face = cv.warpAffine(face, rotation_matrix, (cols, rows))
        transformed_faces.append(rotated_face)

    return transformed_faces


# get the dataset
def get_dataset():
    # load the images
    images = load_images_from_disk(IMAGE_PATHS)
    # load the videos
    videos = load_videos_from_disk(VIDEO_PATHS)
    # combine the images and videos
    dataset = images + videos
    # detect the faces
    detected_faces = detect_faces(dataset)

    return transform_faces(detected_faces)
