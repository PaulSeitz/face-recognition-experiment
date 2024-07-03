import cv2 as cv


def detect_faces_in_image(image, scale_factor=1.2, min_neighbors=5):
    # Load the cascade
    face_cascade_frontal = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Detect faces using the loaded cascade (reduce size by 10% and set the number of needed neighbors to 5)
    faces_front = face_cascade_frontal.detectMultiScale(gray_img, scale_factor, min_neighbors)

    # return the detected faces
    return faces_front


def start_recording():
    # capture video from the webcam
    cap = cv.VideoCapture(0)

    while True:
        current_frame = cap.read()[1]

        if current_frame is None:
            continue

        # Convert the frame to grayscale
        gray_img = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)

        # Detect faces using the loaded cascade (reduce size by 10% and set the number of needed neighbors to 5)
        faces_front = detect_faces_in_image(current_frame)

        # draw rectangles around the detected faces
        for (x, y, w, h) in faces_front:
            cv.rectangle(current_frame, (x, y), (x+w, y+h), (255, 0, 0), 3)

        # Display the output
        cv.imshow('Face Detection', current_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
