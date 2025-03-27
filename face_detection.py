import cv2
import os
import shutil
from deepface import DeepFace

# Directory to store registered faces
REGISTERED_FACES_DIR = "registered_faces"
os.makedirs(REGISTERED_FACES_DIR, exist_ok=True)

# Initialize OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def register_face(username):
    """ Captures an image of the user's face and stores it for recognition """
    cap = cv2.VideoCapture(0)
    print(f"Look at the camera, {username}, and press 's' to save your face.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Face Registration', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Press 's' to save face
            if len(faces) > 0:
                face_image = frame[y:y+h, x:x+w]
                user_folder = os.path.join(REGISTERED_FACES_DIR, username)
                os.makedirs(user_folder, exist_ok=True)

                face_path = os.path.join(user_folder, f"{username}.jpg")
                cv2.imwrite(face_path, face_image)
                print(f"Face saved as {face_path}")
            break
        elif key == ord('q'):  # Press 'q' to cancel
            break

    cap.release()
    cv2.destroyAllWindows()

def recognize_faces():
    """ Real-time recognition of registered faces """
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]

            # Compare face with registered faces
            found = False
            for username in os.listdir(REGISTERED_FACES_DIR):
                user_folder = os.path.join(REGISTERED_FACES_DIR, username)
                user_image_path = os.path.join(user_folder, f"{username}.jpg")

                try:
                    verification = DeepFace.verify(face_region, user_image_path, enforce_detection=False)
                    if verification["verified"]:
                        found = True
                        greeting = f"Hello, {username}"
                        color = (0, 255, 0)  # Green for recognized
                        break
                except Exception as e:
                    print(f"Error verifying {username}: {e}")

            if not found:
                greeting = "Face not recognized"
                color = (0, 0, 255)  # Red for unrecognized

            # Draw rectangle and display name
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, greeting, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main Menu
while True:
    print("\n1️⃣ Register Face\n2️⃣ Start Face Recognition\n3️⃣ Exit")
    choice = input("Enter your choice: ")

    if choice == '1':
        name = input("Enter your username: ")
        register_face(name)
    elif choice == '2':
        recognize_faces()
    elif choice == '3':
        break
    else:
        print("Invalid choice, please try again.")
