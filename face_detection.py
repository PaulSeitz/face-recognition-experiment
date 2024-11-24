import cv2 as cv
import numpy as np
import face_recognition_cnn
import utils
from scipy.fftpack import dct

# "lower" detection threshold, since I prefer to have some false positives come into the recognition step instead of missing a face
# The recognition model will probably filter out most remaining false positives
DETECTION_THRESHOLD = 0.7
FACE_OUTPUT_SIZE = 224  # Common input size for many CNN architectures

# used for debugging
unrecognized_faces = []
# https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
detector = cv.FaceDetectorYN.create('pre_trained_models/face_detection_yunet_2023mar.onnx', "", (1920, 1080), DETECTION_THRESHOLD, 0.3, 5000)


def check_face_quality(face_image, min_size=128, min_variance=1200, blur_threshold=1500):
    """
    Check if a face image is suitable for processing by evaluating:
    1. Image size
    2. Image variance (contrast/detail)
    3. Blur detection using DCT

    Args:
        face_image: numpy array of face image
        min_size: minimum acceptable dimension size
        min_variance: minimum acceptable image variance
        blur_threshold: threshold for blur detection (lower = more strict)

    Returns:
        tuple: (is_acceptable, dict of quality metrics)
    """
    # Check size
    height, width = face_image.shape[:2]
    if height < min_size or width < min_size:
        return False, {"reason": "size too small", "size": (height, width)}

    # Convert to grayscale if needed
    if len(face_image.shape) == 3:
        gray = cv.cvtColor(face_image, cv.COLOR_BGR2GRAY)
    else:
        gray = face_image

    # Check variance (measure of contrast and detail)
    variance = np.var(gray)
    if variance < min_variance:
        return False, {"reason": "low detail/contrast", "variance": variance}

    # Detect blur using DCT
    dct_score = compute_dct_quality(gray)
    if dct_score < blur_threshold:
        return False, {"reason": "too blurry", "blur_score": dct_score}

    return True, {
        "size": (height, width),
        "variance": variance,
        "blur_score": dct_score
    }


def compute_dct_quality(gray_image):
    """
    Compute image quality score using DCT (Discrete Cosine Transform).
    Lower scores indicate more blur.
    """
    # Compute DCT
    dct_array = dct(dct(gray_image.T, norm='ortho').T, norm='ortho')

    # Get absolute values in top-left corner (high frequencies)
    top_left = np.abs(dct_array[:5, :5])

    # Calculate score (higher means sharper image)
    return np.mean(top_left)

# show images
def show_images(image: np.ndarray):
    cv.imshow('Image', image)
    # wait for the esc key to be pressed
    if cv.waitKey(0) & 0xFF == 27:
        return

    cv.destroyAllWindows()


def calculate_face_snippet(face, width, height):
    """
    Calculate a snippet around the face with a small margin.
    :param face: image coordinates of the face
    :param width: dimensions of the image
    :param height: dimensions of the image
    :return: the coordinates of the bounding box
    """
    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])  # rest are landmark coordinates, not relevant here
    # calculate the margin
    margin_w, margin_h = int(0.1 * w), int(0.1 * h)
    # calculate the image coordinates
    x1, y1 = max(0, x - margin_w), max(0, y - margin_h)
    x2, y2 = min(width, x + w + margin_w), min(height, y + h + margin_h)
    return x1, y1, x2, y2

def draw_landmarks(image, face):
    """
    Draw landmarks on the image. This includes:
    - Eye centers (2)
    - Nose tip (1)
    - Mouth corners (2)
    TODO: Add more landmarks for better visualization
    :param image: the complete image
    :param face: coordinates of the relevant landmarks that were extracted by the face detector, the relevant landmarks are 4-14
    """
    landmarks = face[4:14].reshape(-1, 2)
    for i, (x, y) in enumerate(landmarks):
        cv.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
        cv.putText(image, str(i+1), (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)


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
    """
    Detect faces in an image and extract them for recognition.
    :param image: the image to process
    :return: a list of extracted faces and the visualization image with all bounding boxes and landmarks
    """
    height, width = image.shape[:2]
    detector.setInputSize((width, height))

    # Keep BGR format for detection
    faces_image = detector.detect(image)
    extracted_faces = []
    visualization_image = image.copy()

    if faces_image[1] is None:
        return [], visualization_image

    for idx, face in enumerate(faces_image[1]):
        # Align face while still in BGR format
        aligned_face = align_and_resize_face(image, face)

        if aligned_face is not None and isinstance(aligned_face, np.ndarray):
            extracted_faces.append(aligned_face)

        # Draw visualization (using original BGR image)
        x, y, w, h = face[:4].astype(int)
        cv.rectangle(visualization_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add face index
        cv.putText(visualization_image, f"Face {idx + 1}", (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        landmarks = face[4:14].reshape(-1, 2).astype(int)
        for i, (lx, ly) in enumerate(landmarks):
            cv.circle(visualization_image, (lx, ly), 2, (0, 255, 0), -1)
            cv.putText(visualization_image, str(i + 1), (lx, ly),
                       cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    return extracted_faces, visualization_image


def start_recording(model_path, device):
    """
    Start the real-time face detection and recognition pipeline.
    :param model_path: path to the model that should be used for the recognition
    :param device: the device to process the recognition (i.e. cpu or gpu)
    """

    # initialize the model
    model = face_recognition_cnn.FaceRecognitionCNN({}, device)
    model.load_model(model_path, device)
    model.to(device)

    # initialize the camera (uses the default camera)
    cap = cv.VideoCapture(0)
    cap.set(3, 1024)
    cap.set(4, 1024)
    frame_counter = 0

    # start the live feed and process frame by frame
    while True:
        # stop on q
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # only process every 10th frame to reduce load
        if frame_counter < 10:
            frame_counter += 1
            continue

        ret, current_frame = cap.read()
        if not ret:
            continue

        # extract the faces from a frame and save the image with drawn in bounding boxes and landmarks
        extracted_faces, visualization_frame = detect_faces_in_image(current_frame)

        # process each face
        for i, face in enumerate(extracted_faces):
            # perform the same preprocessing pipeline as during training to ensure consistency
            face_tensor = utils.resize_image(face)
            face_tensor = face_tensor.unsqueeze(0)

            # predict the label and confidence
            label, confidence = model.predict(face_tensor, device)

            # filter out faces that are too small / too pixelated due to being too small
            is_acceptable, quality_metrics = check_face_quality(face)

            if is_acceptable:
                # Add label to visualization frame
                cv.putText(visualization_frame, f"Face {i + 1}: {label} ({confidence})", (10, 30 + i * 30), cv.FONT_HERSHEY_SIMPLEX, 0.9,
                           (0, 0, 255), 2)
            else:
                # Add label to visualization frame
                cv.putText(visualization_frame, f"Face {i + 1}: Unrecognized, Reason: {quality_metrics}", (10, 30 + i * 30), cv.FONT_HERSHEY_SIMPLEX, 0.9,
                           (0, 0, 255), 2)
                unrecognized_faces.append(face)

        # show the visualization frame
        cv.imshow('Face Detection and Recognition', visualization_frame)
        frame_counter = 0

    # release the camera and close the window
    cap.release()
    cv.destroyAllWindows()