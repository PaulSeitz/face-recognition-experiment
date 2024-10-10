import cv2 as cv
import numpy as np
import face_recognition_cnn
import utils

# "lower" detection threshold, since I prefer to have some false positives come into the recognition step instead of missing a face
# The recognition model will probably filter out most remaining false positives
DETECTION_THRESHOLD = 0.7
FACE_OUTPUT_SIZE = 224  # Common input size for many CNN architectures

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
    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])  # rest are landmark coordinates, not relevant here
    # calculate the margin
    margin_w, margin_h = int(0.1 * w), int(0.1 * h)
    # calculate the image coordinates
    x1, y1 = max(0, x - margin_w), max(0, y - margin_h)
    x2, y2 = min(width, x + w + margin_w), min(height, y + h + margin_h)
    return x1, y1, x2, y2

def draw_landmarks(image, face):
    landmarks = face[4:14].reshape(-1, 2)
    for i, (x, y) in enumerate(landmarks):
        cv.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
        cv.putText(image, str(i+1), (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)


detector = cv.FaceDetectorYN.create('pre_trained_models/face_detection_yunet_2023mar.onnx', "", (1920, 1080),
                                    DETECTION_THRESHOLD, 0.3, 5000)


def align_and_resize_face(image, face, output_size=FACE_OUTPUT_SIZE):
    """
    Align the face and resize it to a consistent output size.

    :param image: Input image containing the face (RGB)
    :param face: Face information from the detector, including bounding box and landmarks
    :param output_size: Desired width and height of the output face image
    :return: Aligned and resized face image
    """
    x, y, w, h = face[:4].astype(float)
    landmarks = face[4:14].reshape(-1, 2)

    # Calculate the center of the eyes
    left_eye_center, right_eye_center = landmarks[:2]
    eye_center = ((left_eye_center + right_eye_center) / 2).astype(float)

    # Calculate the angle between the eye centers
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # Calculate the scale
    desiredEyeDistance = 0.3 * output_size
    eyeDistance = np.sqrt((dX ** 2) + (dY ** 2))
    scale = desiredEyeDistance / eyeDistance

    # Calculate rotation matrix
    M = cv.getRotationMatrix2D(tuple(eye_center), angle, scale)

    # Update the translation component of the matrix
    tX = output_size * 0.5 - eye_center[0]
    tY = output_size * 0.35 - eye_center[1]  # Place eyes at 35% from the top
    M[0, 2] += tX
    M[1, 2] += tY

    # Apply the affine transformation
    output = cv.warpAffine(image, M, (output_size, output_size),
                           flags=cv.INTER_CUBIC)

    return output


def detect_faces_in_image(image):
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    faces_image = detector.detect(image_rgb)

    extracted_faces = []
    visualization_image = image.copy()

    if faces_image[1] is None:
        return [], visualization_image

    for face in faces_image[1]:
        # Align and resize the face
        aligned_face = align_and_resize_face(image, face)
        extracted_faces.append(aligned_face)

        # Draw bounding box
        x, y, w, h = face[:4].astype(int)
        cv.rectangle(visualization_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw landmarks
        landmarks = face[4:14].reshape(-1, 2).astype(int)
        for i, (lx, ly) in enumerate(landmarks):
            cv.circle(visualization_image, (lx, ly), 2, (0, 255, 0), -1)
            cv.putText(visualization_image, str(i + 1), (lx, ly), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    return extracted_faces, visualization_image


def start_recording(model_path, device):
    model = face_recognition_cnn.FaceRecognitionCNN(None, device, load_model=model_path)
    model.to(device)

    cap = cv.VideoCapture(0)
    cap.set(3, 1024)
    cap.set(4, 1024)
    frame_counter = 0

    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_counter < 10:
            frame_counter += 1
            continue

        ret, current_frame = cap.read()
        if not ret:
            continue

        extracted_faces, visualization_frame = detect_faces_in_image(current_frame)

        for i, face in enumerate(extracted_faces):
            face_tensor = utils.resize_image(face)  # Ensure this function returns the correct tensor format
            face_tensor = face_tensor.unsqueeze(0)

            label = model.predict(face_tensor, device)

            # Add label to visualization frame
            cv.putText(visualization_frame, f"Face {i + 1}: {label}", (10, 30 + i * 30), cv.FONT_HERSHEY_SIMPLEX, 0.9,
                       (0, 0, 255), 2)

        cv.imshow('Face Detection and Recognition', visualization_frame)
        frame_counter = 0

    cap.release()
    cv.destroyAllWindows()