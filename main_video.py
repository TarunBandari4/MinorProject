import cv2
import threading
from simple_facerec import SimpleFacerec
from queue import Queue

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load Camera
cap = cv2.VideoCapture(0)

# Thread-safe queue for frames
frame_queue = Queue(maxsize=1)
output_frame = None

# Flag to signal when to stop threads
stop_thread = False
camera_on = True  # Flag to toggle the camera

# Variable to store the recognized face name
recognized_face_name = "Unknown Face"


def capture_frames():
    global stop_thread
    while not stop_thread:
        if camera_on:
            ret, frame = cap.read()
            if not ret:
                continue
            if not frame_queue.full():
                frame_queue.put(frame)


def process_frames():
    global stop_thread, output_frame, recognized_face_name
    while not stop_thread:
        if camera_on and not frame_queue.empty():
            frame = frame_queue.get()
            # Detect Faces
            face_locations, face_names = sfr.detect_known_faces(frame)
            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

                # Update the recognized face name
                recognized_face_name = name

            # If no face is recognized, show the last recognized name
            output_frame = frame
            if recognized_face_name != "Unknown":
                cv2.putText(output_frame, recognized_face_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        elif not camera_on:
            # Reset output frame when camera is off
            output_frame = None


# Start threads for capturing and processing
capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)

capture_thread.start()
process_thread.start()

while True:
    if output_frame is not None:
        cv2.imshow("Frame", output_frame)

    key = cv2.waitKey(1)

    # ESC key to stop, 'c' key to toggle camera on/off
    if key == 27:  # ESC key to stop
        stop_thread = True
        break
    elif key == ord('c'):  # Toggle camera with 'c' key
        camera_on = not camera_on
        if camera_on:
            print("Camera turned on.")
        else:
            print("Camera turned off.")

# Wait for threads to finish
capture_thread.join()
process_thread.join()

cap.release()
cv2.destroyAllWindows()

# Print the recognized face name or "Unknown Face"
if recognized_face_name:
    print(f"Recognized Face: {recognized_face_name}" if recognized_face_name != "Unknown" else "Unknown Face")
else:
    print("No face recognized.")
