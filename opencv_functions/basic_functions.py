import cv2
import numpy as np


def resize_image(img: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    """
    Resizes an image to a specified width and height.

    Args:
        img: A NumPy array representing the image.
        new_width: The desired width of the resized image.
        new_height: The desired height of the resized image.

    Returns:
        A NumPy array representing the resized image.
    """
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """Converts a BGR image to grayscale.

    Args:
        img: A numpy array representing a BGR image.

    Returns:
        A numpy array representing the grayscale version of the input image.
    """
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return grayscale_image


def flip_image(
    img: np.ndarray, flip_horizontal: bool = True, flip_vertical: bool = False
) -> np.ndarray:
    """
    Flips an image along a specified axis.

    Args:
        img: A NumPy array representing the image.
        flip_horizontal: Flag indicating whether to flip horizontally
                         (left-right). Defaults to True.
        flip_vertical: Flag indicating whether to flip vertically (up-down).
                       Defaults to False.

    Returns:
        A NumPy array representing the flipped image.

    This function flips the image based on a combination of the
    flip_horizontal and flip_vertical flags.
    """

    # Validate flags: Ensure only one flipping direction is active
    if flip_horizontal and flip_vertical:
        flipped_img = cv2.flip(img, -1)  # Horizontal and vertical
    elif flip_horizontal:
        flipped_img = cv2.flip(img, 0)  # Vertical flip
    elif flip_vertical:
        flipped_img = cv2.flip(img, 1)  # Vertical and Horizontal flip

    return flipped_img


def rotate_image_multiple_of_90(img: np.ndarray, angle: int = 90) -> np.ndarray:
    """
    Rotates an image by a specified angle of 90, 180 or 280 degrees.

    Args:
        img: A NumPy array representing the image.
        angle: The rotation angle in degrees (clockwise). Defaults to 90 degrees.

    Returns:
        A NumPy array representing the rotated image.

    This function rotates the image clockwise by the specified angle.

    Tips:
        - Explore OpenCV functionalities like cv2.getRotationMatrix2D and cv2.warpAffine
          to rotate image to any desired angle
        - Consider incorporating options for specifying rotation direction (clockwise or counter-clockwise).
    """

    # Rotate image based on angle
    if angle == 90:
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        rotated_img = cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # Handle unsupported angles or provide a default behavior (e.g., raise an error)
        raise ValueError(
            f"Unsupported rotation angle: {angle}. Currently supports multiples of 90 degrees."
        )

    return rotated_img


def crop_image(
    img: np.ndarray,
    top_left_x: int,
    top_left_y: int,
    bottom_right_x: int,
    bottom_right_y: int,
) -> np.ndarray:
    """
    Crops a rectangular region from an image.

    Args:
        img: A NumPy array representing the image.
        top_left_x: The X coordinate of the top-left corner of the cropping
                    region.
        top_left_y: The Y coordinate of the top-left corner of the cropping
                    region.
        bottom_right_x: The X coordinate of the bottom-right corner of the
                        cropping region (inclusive).
        bottom_right_y: The Y coordinate of the bottom-right corner of the
                        cropping region (inclusive).

    Returns:
        A NumPy array representing the cropped image.

    This function crops a rectangular region from the image defined by the
    provided coordinates.
    """

    # Extract the cropping region using NumPy slicing
    cropped_img = img[top_left_y : bottom_right_y + 1, top_left_x : bottom_right_x + 1]

    return cropped_img


def add_text_to_image(img: np.ndarray, text: str) -> np.ndarray:
    """
    Adds text to an image.

    Args:
    img: A NumPy array representing the image.
    text: The text to be added.

    Returns:
    A NumPy array representing the image with text added.

    This function does not modify the original image and returns a new image with text.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    return cv2.putText(img, text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)


def modify_pixel_value(
    img: np.ndarray,
    new_r: int,
    new_g: int,
    new_b: int,
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
) -> np.ndarray:
    """
    Modifies the pixel values within a rectangular area in a BGR image.

    Args:
        img: A NumPy array representing the BGR image.
        new_r: The new red value for the pixels (0-255).
        new_g: The new green value for the pixels (0-255).
        new_b: The new blue value for the pixels (0-255).
        start_x: The X coordinate of the top-left corner of the area.
        start_y: The Y coordinate of the top-left corner of the area.
        end_x: The X coordinate (exclusive) of the bottom-right corner of the area + 1.
        end_y: The Y coordinate (exclusive) of the bottom-right corner of the area + 1.

    Returns:
        A NumPy array representing the modified image with the updated pixel values.

    This function modifies the original image in-place and returns it.
    """

    img[start_y:end_y, start_x:end_x] = [
        new_b,
        new_g,
        new_r,
    ]  # Set all pixels to new color
    return img
