import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

# Open the camera
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Load known faces
known_face_encodings = []
known_face_names = []

face_files = {
    "alia": "photos/alia.jpg",
    "badshah": "photos/badshah.jpg",
    "deepika": "photos/deepika.jpg",
    "elvish": "photos/elvish.jpg",
    "honey": "photos/honey.jpg",
    "mona": "photos/mona.jpg",
    "ranveer": "photos/ranveer.jpg",
    "ratan": "photos/ratan.jpg",
    "shubh": "photos/shubh.jpg",
    "stev": "photos/stev.jpg"
}

for name, filepath in face_files.items():
    if os.path.exists(filepath):
        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(name)
        else:
            print(f"Warning: No face found in {filepath}")
    else:
        print(f"Error: File {filepath} not found.")

students = known_face_names.copy()
face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Ensure the directory exists
os.makedirs("attendance_records", exist_ok=True)
csv_filename = os.path.join("attendance_records", f"{current_date}.csv")

# Keep track of already marked students
marked_students = set()

try:
    with open(csv_filename, 'a', newline='') as f:  # Changed to 'a' mode (append)
        lnwriter = csv.writer(f)

        # If the file is new, write headers
        if os.stat(csv_filename).st_size == 0:
            lnwriter.writerow(["Name", "Time"])

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Failed to capture frame from camera.")
                break

            cv2.imshow("Attendance System", frame)

            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            if s:
                face_locations = face_recognition.face_locations(rgb_small_frame)

                if face_locations:
                    print(f"Detected {len(face_locations)} face(s).")

                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    face_names = []

                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        name = "Unknown"

                        if True in matches:
                            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)

                            if matches[best_match_index]:
                                name = known_face_names[best_match_index]

                        face_names.append(name)

                        if name in students and name not in marked_students:
                            marked_students.add(name)  # Prevent duplicate marking
                            current_time = datetime.now().strftime("%H:%M:%S")
                            lnwriter.writerow([name, current_time])
                            f.flush()  # Ensure data is written immediately
                            print(f"{name} marked present at {current_time}")

                else:
                    print("No faces detected.")

            # Press 'q' to quit
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:  # 27 is the ESC key
                print("Exiting program...")
                break

finally:
    video_capture.release()
    cv2.destroyAllWindows()
    print("Attendance marked successfully.")
