import cv2
import os
import numpy as np
from deepface import DeepFace

# Folder to store registered faces
REGISTERED_FACES_DIR = "registered_faces"
os.makedirs(REGISTERED_FACES_DIR, exist_ok=True)

# Initialize OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def save_face(frame, face, username):
    """ Save detected face as an image """
    x, y, w, h = face
    face_img = frame[y:y+h, x:x+w]
    user_folder = os.path.join(REGISTERED_FACES_DIR, username)
    os.makedirs(user_folder, exist_ok=True)

    face_path = os.path.join(user_folder, f"{username}.jpg")
    cv2.imwrite(face_path, face_img)
    print(f"Face registered for {username}!")

def recognize_faces():
    """ Detects and recognizes faces in real-time """
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]

            recognized = False
            for username in os.listdir(REGISTERED_FACES_DIR):
                user_folder = os.path.join(REGISTERED_FACES_DIR, username)
                user_image_path = os.path.join(user_folder, f"{username}.jpg")

                try:
                    result = DeepFace.verify(face_region, user_image_path, enforce_detection=False)
                    if result["verified"]:
                        recognized = True
                        greeting = f"Hello, {username}! Welcome!"
                        color = (0, 255, 0)  # Green
                        break
                except:
                    pass

            if not recognized:
                color = (0, 0, 255)  # Red
                greeting = "New face detected! Press 'r' to register."

            # Draw rectangle and display text
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, greeting, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):  # Register new face
            if len(faces) > 0:
                username = input("Enter username: ")
                save_face(frame, faces[0], username)

        elif key == ord('q'):  # Quit
            break

    cap.release()
    cv2.destroyAllWindows()

# Start real-time face recognition
recognize_faces()
