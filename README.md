# Image Processing with OpenCV

This project provides a set of basic and advanced image processing operations using OpenCV, a popular computer vision library.

Additionally you can find a Gender and Age detection model implementatin in the folder "gender_and_age_detection"

## Image Processing Operations

### Basic Operations

- `RESIZE_IMAGE`: Resizes the image to the specified dimensions.
- `CONVERT_TO_GRAYSCALE`: Converts the image to grayscale.
- `ADD_TEXT_TO_IMAGE`: Adds the specified text to the image.
- `MODIFY_PIXEL_VALUE`: Modifies the RGB values of the pixels in the specified region.
- `FLIP_IMAGE`: Flips the image horizontally and/or vertically.
- `ROTATE_IMAGE_MULTIPLE_90`: Rotates the image by the specified multiple of 90 degrees.
- `CROP_IMAGE`: Crops the image to the specified region.

### Advanced Operatinos

- `EDGE_DETECTION`: Detects the edges in the image using the specified algorithm (e.g., "Sobel").
- `CHANGE_COLOR_SPACE`: Converts the image to the specified color space (e.g., "LAB").
- `ORB_FEATURE_DETECTOR`: Detects the Oriented FAST and Rotated BRIEF (ORB) features in the image.
- `MATCH_KEY_POINTS`: Matches the ORB features between two images and displays the matches.

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

Install OpenCV and NumPy using pip within the activated virtual environment:

```
pip install -r requirements.txts
```

## Usage

1. Open the `config.py` file and replace `ORIGINAL_IMAGE_PATH` and `IMAGE_2_PATH` (if required) in the code with the actual paths to your images.
2. Use the `config.py` file to select wich opencv functionality you want to test. Just modife the `False` value to `True` for your desired functionality.
   **Warning!: Just one of the values should be True in order to display the output correctly**
3. Run the scrip:
   ```
   python main.py
   ```

The script will first display the original image. Then, it will perform a set of basic and/or advanced image processing operations, depending on the values of the constants defined at the top of the script. The modified image will then be displayed and saved to OUTPUT_IMAGE_PATH.
