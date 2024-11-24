import argparse
from face_recognition_cnn import FaceRecognitionCNN
from face_detection import detect_faces_in_image, check_face_quality
import utils
import cv2 as cv

DEVICE = "cpu"

if __name__ == "__main__":
    # get the image path from the command line arguments
    parser = argparse.ArgumentParser(description='Real time face detection and recognition pipeline')
    parser.add_argument('--image', type=str, help='Path to the image to process', required=True)
    parser.add_argument('--model', type=str, help='Path to the model to use for face recognition', required=True)

    args = parser.parse_args()

    # load the image from disk
    current_frame = cv.imread(args.image)

    # Load the model
    model = FaceRecognitionCNN({}, DEVICE)
    model.load_model(args.model, DEVICE)

    extracted_faces, visualization_frame = detect_faces_in_image(current_frame)

    for i, face in enumerate(extracted_faces):
        face_tensor = utils.resize_image(face)

        face_tensor = face_tensor.unsqueeze(0)

        label, confidence = model.predict(face_tensor, DEVICE)

        # filter out faces that are too small / too pixelated due to being too small
        is_acceptable, quality_metrics = check_face_quality(face)

        if is_acceptable:
            # Add label to visualization frame
            cv.putText(visualization_frame, f"Face {i + 1}: {label} ({confidence})", (10, 30 + i * 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.9,
                       (0, 0, 255), 2)
        else:
            # Add label to visualization frame
            cv.putText(visualization_frame, f"Face {i + 1}: Unrecognized, Reason: {quality_metrics}", (10, 30 + i * 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.9,
                       (0, 0, 255), 2)

    cv.imshow('Face Detection and Recognition', visualization_frame)

    cv.waitKey(0)