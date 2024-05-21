# Gender and Age Detection with OpenCV

This project implements real-time gender and age detection using pre-trained deep learning models and OpenCV.

## Requirements

Python (tested with version 3.10.0)
OpenCV (tested with version 4.9.0.80)
NumPy (tested with version 1.26.4)

## Create a Virtual Environment (Optional)

There are two main ways to create a virtual environment depending on your operating system:

Using venv module (Python 3.3+):

1. Open a terminal or command prompt.

2. Navigate to the directory where you want to create your project.

3. Create a virtual environment named `myenv` using the following command:

   ```
   python -m venv myenv
   ```

   Replace `myenv` with your desired name for the virtual environment.

4. Activate the virtual environment:
   Linux/macOS:

   ```
   source myenv/bin/activate
   ```

   Windows:

   ```
   myenv\Scripts\activate
   ```

## Install packages

**Once your virtual environment is activated, you can proceed with installing the project dependencies:**

1. Install OpenCV and NumPy using pip within the activated virtual environment:

   ```
   pip install opencv-python numpy
   ```

## Usage

1. Run the main script:

   ```
   python main.py
   ```

2. The script will open your webcam and display the video feed with detected faces, predicted age range, and predicted gender. Press 'q' to quit the program.

**Remember to activate your virtual environment before running the script.**

## Code Structure

- `gender_and_age_detection.py` (empty): This file serves as the main project directory.
- `models`: This folder stores all the pre-trained models used by the script.
- `main.py`: This script performs the following tasks:
  - Loads pre-trained models (defined in `main.py`).
  - Opens the webcam video capture.
  - Processes each video frame:
    - Calls the detect_faces_in_frame function (defined in utils.py) to detect faces.
    - For each detected face:
      - Extracts the face region from the frame.
      - Predicts the gender and age range using the pre-trained models.
      - Draws bounding boxes and labels (predicted age and gender) around the detected faces on the frame.
  - Displays the processed video frame with labels and bounding boxes.
- `utils.py`: This file contains utility functions, including:
  - `detect_faces_in_frame`: This function detects faces in a frame using a pre-trained deep learning network and returns the frame with highlights and a list of bounding boxes for detected faces.

## Explanation of `utils.py`:

The `detect_faces_in_frame` function in `utils.py` performs the core face detection functionality. It takes three arguments:

`face_detection_net`: The pre-trained deep learning network for face detection.
`frame`: The input frame as a NumPy array (assumed to be in BGR format).
`confidence_threshold` (optional): Confidence threshold for filtering detections (default: 0.7).

The function performs the following steps:

1. Creates a copy of the frame to avoid modifying the original.
2. Extracts frame dimensions (height and width).
3. Creates a blob from the frame for feeding into the network (performs necessary pre-processing).
4. Sets the network input with the created blob.
5. Performs a forward pass to get the network predictions for face detection.
6. Initializes an empty list to store detected face bounding boxes.
7. Iterates through each detection and extracts relevant information:
   - Confidence score for the detection.
   - Filters out detections with low confidence below the specified threshold.
   - Extracts bounding box coordinates for the detected face.
8. Appends bounding box coordinates to the list.
9. Draws green rectangles around the detected faces on the frame copy.
10. Returns the frame with highlights and the list of detected face bounding boxes.
