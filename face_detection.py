import cv2 as cv
import numpy as np
import face_recognition_cnn
import dataloader

# "lower" detection threshold, since I prefer to have some false positives come into the recognition step instead of missing a face
# The recognition model will probably filter out most remaining false positives
DETECTION_THRESHOLD = 0.7

# used for debugging
unrecognized_faces = []
# https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
detector = cv.FaceDetectorYN.create('pre_trained_models/face_detection_yunet_2023mar.onnx', "", (1920, 1080), DETECTION_THRESHOLD, 0.3, 5000)


# show images
def show_images(image: np.ndarray):
    cv.imshow('Image', image)
    # wait for the esc key to be pressed
    if cv.waitKey(0) & 0xFF == 27:
        return

    cv.destroyAllWindows()


def calculate_face_snippet(face, width, height):
    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])  # rest are landmark coordinates, not relevant for me
    # calculate the margin
    margin_w, margin_h = int(0.1 * w), int(0.1 * h)
    # calculate the image coordinates
    x1, y1 = max(0, x - margin_w), max(0, y - margin_h)
    x2, y2 = min(width, x + w + margin_w), min(height, y + h + margin_h)
    return x1, y1, x2, y2


def detect_faces_in_image(image):

    # Get image dimensions
    height, width = image.shape[:2]

    detector.setInputSize((width, height))

    # transform to black and white
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    faces_image = detector.detect(image)

    # List to store the extracted face images
    extracted_faces: list[np.ndarray] = []

    # check if anything was found
    if faces_image[1] is None:
        unrecognized_faces.append(np.array(image))
        return

    # Extract each detected face
    for idx, face in enumerate(faces_image[1]):
        x1, y1, x2, y2 = calculate_face_snippet(face, width, height)

        # extract the face
        face_img = image[y1:y2, x1:x2]

        extracted_faces.append(face_img)

    if not extracted_faces:
        unrecognized_faces.append(np.array(image))

    # Return the list of extracted face images
    return extracted_faces


def start_recording(model_path, device):

    # load the model
    model = face_recognition_cnn.FaceRecognitionCNN(None, device, load_model=model_path)
    model.to(device)

    # capture video from the webcam
    cap = cv.VideoCapture(0)
    cap.set(3, 1024)
    cap.set(4, 1024)
    frame_counter = 0
    while True:

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # only process every 10th frame
        if frame_counter < 10:
            frame_counter += 1
            continue

        current_frame = cap.read()[1]

        if current_frame is None:
            continue

        height, width = current_frame.shape[:2]
        detector.setInputSize((width, height))

        # Detect faces using the detector
        faces = detector.detect(current_frame)

        if faces[1] is None:
            cv.imshow('Face Detection', current_frame)
            continue

        # draw rectangles around the detected faces
        for idx, face in enumerate(faces[1]):
            x1, y1, x2, y2 = calculate_face_snippet(face, width, height)

            # draw the rectangle
            cv.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # extract the face
            face_img = current_frame[y1:y2, x1:x2]

            face_tensor = dataloader.resize_image(face_img)

            face_tensor = face_tensor.unsqueeze(0)

            # try to recognize the face
            label = model.predict(face_tensor, device)

            # write the label in red above the face
            cv.putText(current_frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the output
        cv.imshow('Face Detection', current_frame)

        frame_counter = 0

    cap.release()
    cv.destroyAllWindows()
