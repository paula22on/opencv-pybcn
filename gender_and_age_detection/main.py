import cv2

from utils import detect_faces_in_frame

print("Initializing variables...")
# Define paths to pre-trained models
face_detection_prototxt = "models/opencv_face_detector.pbtxt"
face_detection_model = "models/opencv_face_detector_uint8.pb"
age_estimation_prototxt = "models/age_deploy.prototxt"
age_estimation_model = "models/age_net.caffemodel"
gender_classification_prototxt = "models/gender_deploy.prototxt"
gender_classification_model = "models/gender_net.caffemodel"

# Define the mean values used for pre-processing images
model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)

# Define labels for the age prediction output
age_labels = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)",
]

# Define labels for the gender prediction output
gender_labels = ["Male", "Female"]

# Load the pre-trained models
print("Loading pretrained models...")
face_net = cv2.dnn.readNet(face_detection_model, face_detection_prototxt)
age_net = cv2.dnn.readNet(age_estimation_model, age_estimation_prototxt)
gender_net = cv2.dnn.readNet(
    gender_classification_model, gender_classification_prototxt
)

print("Loading pretrained models...")
# Open video capture (0 for webcam, or path to video file)
video_capture = cv2.VideoCapture(0)

# Padding for extracting the face region around the bounding box
face_extraction_padding = 20

while True:
    # Check if frame is read successfully
    has_frame, frame = video_capture.read()
    if not has_frame:
        cv2.waitKey()
        break

    # Detect faces in the frame
    result_image, face_boxes = detect_faces_in_frame(face_net, frame)

    # If no faces were detected, inform the user and continue to the next frame
    if not face_boxes:
        print("No face detected")
        continue

    # Loop through each detected face bounding box
    for face_box in face_boxes:
        # Extract the face region from the original frame based on the bounding box
        face = frame[
            max(0, face_box[1] - face_extraction_padding) : min(
                face_box[3] + face_extraction_padding, frame.shape[0] - 1
            ),
            max(0, face_box[0] - face_extraction_padding) : min(
                face_box[2] + face_extraction_padding, frame.shape[1] - 1
            ),
        ]

        # Create a blob from the extracted face region for feeding into the network
        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), model_mean_values, swapRB=False
        )

        # Set the network input for gender prediction
        gender_net.setInput(blob)
        gender_predictions = gender_net.forward()

        # Get the predicted gender label (index with highest probability)
        predicted_gender = gender_labels[gender_predictions[0].argmax()]
        print(f"Gender: {predicted_gender}")

        # Set the network input for age prediction (using the same blob)
        age_net.setInput(blob)
        age_predictions = age_net.forward()

        # Get the predicted age range label (index with highest probability)
        predicted_age = age_labels[age_predictions[0].argmax()]
        print(f"Age: {predicted_age[1:-1]} years")

        # Draw text labels (gender and age) on the frame with the detected face
        cv2.putText(
            result_image,
            f"{predicted_gender}, {predicted_age}",
            (face_box[0], face_box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Display the resulting frame with highlighted faces and labels
        cv2.imshow("Detecting age and gender", result_image)

    # Wait for a key press. 'q' to quit
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
