import cv2

from config import (
    ADD_TEXT_TO_IMAGE,
    CHANGE_COLOR_SPACE,
    CROP_IMAGE,
    EDGE_DETECTION,
    FLIP_IMAGE,
    MODIFY_PIXEL_VALUE,
    ORIGINAL_IMAGE_PATH,
    OUTPUT_IMAGE_PATH,
    RESIZE_IMAGE,
    ROTATE_IMAGE_MULTIPLE_90,
)
from opencv_functions.advanced_functions import change_color_space, edge_detection
from opencv_functions.basic_functions import (
    add_text_to_image,
    convert_to_grayscale,
    crop_image,
    flip_image,
    modify_pixel_value,
    resize_image,
    rotate_image_multiple_of_90,
)

# Load an image (replace "path/to/image.jpg" with your actual image path)
img = cv2.imread(ORIGINAL_IMAGE_PATH)

# Check if image loaded successfully
if img is None:
    print("Error: Could not read image!")
    exit(1)

# Image properties
image_size = img.shape
image_type = img.dtype

print("Image size:", image_size)
print("Image data type:", image_type)

# Display the original image
cv2.imshow("Original Image", img)
cv2.waitKey(0)  # Wait for a key press to close the window

# Basic Image Operations Using OpenCV
if RESIZE_IMAGE:
    modified_image = resize_image(img, new_width=500, new_height=400)

if CHANGE_COLOR_SPACE:
    modified_image = change_color_space(img, color_space="LAB")

if ADD_TEXT_TO_IMAGE:
    modified_image = add_text_to_image(img, text="Hi! :)")

if MODIFY_PIXEL_VALUE:
    modified_image = modify_pixel_value(
        img,
        new_r=255,
        new_g=0,
        new_b=0,
        start_x=100,
        start_y=100,
        end_x=200,
        end_y=200,
    )

if FLIP_IMAGE:
    modified_image = flip_image(img, flip_horizontal=True, flip_vertical=True)

if ROTATE_IMAGE_MULTIPLE_90:
    modified_image = rotate_image_multiple_of_90(img, angle=180)

if CROP_IMAGE:
    modified_image = crop_image(
        img,
        top_left_x=300,
        top_left_y=300,
        bottom_right_x=700,
        bottom_right_y=700,
    )

# Advanced Image Operations Using OpenCV
if EDGE_DETECTION:
    modified_image = edge_detection(img, algorithm="Sobel")


# Display modified image
cv2.imshow("Modified Image", modified_image)
cv2.waitKey(0)


# Save the modified image TODO: ADD A GENERIC PATH
cv2.imwrite(OUTPUT_IMAGE_PATH, modified_image)
print("Image saved as modified_image.jpg")

# Close all windows
cv2.destroyAllWindows()
