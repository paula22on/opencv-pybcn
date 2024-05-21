import cv2


def detect_faces_in_frame(face_detection_net, frame, confidence_threshold=0.7):
    """
    This function detects faces in a frame using a pre-trained deep learning network
    and highlights them with a green rectangle.

    Args:
        face_detection_net (cv2.dnn.Net): The pre-trained deep learning network for face detection.
        frame (np.ndarray): The input frame as a NumPy array (assumed to be in BGR format).
        confidence_threshold (float, optional): Confidence threshold for filtering detections (default: 0.7).

    Returns:
        tuple: A tuple containing two elements:
            - frame_with_highlights (np.ndarray): The frame copy with highlighted faces (if any were detected).
            - detected_face_boxes (list): A list containing the coordinates of the detected faces (empty if none were found).
    """

    # Create a copy of the frame to avoid modifying the original
    frame_with_highlights = frame.copy()

    # Get frame dimensions (height and width)
    frame_height = frame_with_highlights.shape[0]
    frame_width = frame_with_highlights.shape[1]

    # Create a blob from the frame for feeding into the network
    blob = cv2.dnn.blobFromImage(
        frame_with_highlights,
        1.0,  # Scale factor
        (300, 300),  # Target size for the network
        [104, 117, 123],  # Mean subtraction (BGR)
        swapRB=True,  # Swap channels from BGR to RGB (if needed by network)
        crop=False,  # Don't crop the image
    )

    # Set the network input
    face_detection_net.setInput(blob)

    # Perform a forward pass to get the network predictions
    detections = face_detection_net.forward()

    # Initialize an empty list to store face bounding boxes
    detected_face_boxes = []

    # Loop over each detection and extract relevant information
    for i in range(detections.shape[2]):
        confidence_score = detections[0, 0, i, 2]  # Confidence score for this detection

        # Filter out detections with low confidence
        if confidence_score > confidence_threshold:
            # Extract bounding box coordinates for the face
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)

            # Append bounding box coordinates to the list
            detected_face_boxes.append([x1, y1, x2, y2])

            # Draw a green rectangle around the detected face
            cv2.rectangle(
                frame_with_highlights,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                int(round(frame_height / 150)),
                8,
            )

    # Return the frame with highlighted faces and the list of face bounding boxes
    return frame_with_highlights, detected_face_boxes
