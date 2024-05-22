import cv2
import numpy as np


def edge_detection(img, algorithm="Canny"):
    """
    Detects edges of an image using either Sobel or Canny edge detection
    algorithms.

    Args:
        img (numpy.ndarray): A NumPy array representing the image in BGR color
                             format.

    Returns:
        numpy.ndarray: A NumPy array representing the image with the edges
                       detected.

    Raises:
        ValueError: If the provided algorithm is not 'Sobel' or 'Canny'.
    """

    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    if algorithm == "Sobel":
        # Sobel Edge Detection
        edges = cv2.Sobel(
            src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5
        )  # Combined X and Y Sobel Edge Detection

        return edges

    if algorithm == "Canny":
        edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

        return edges

    else:
        raise ValueError(
            f"Invalid algorithm: {algorithm}. Supported options are 'Sobel' and 'Canny'."
        )


def change_color_space(img: np.ndarray, color_space: str) -> np.ndarray:
    """
    Converts an image to a specified color space.

    Args:
        img: A NumPy array representing the image (assumed to be in BGR format).
        color_space: The target color space (e.g., "HSV", "LAB", "HSL").

    Returns:
        A NumPy array representing the image in the new color space.

    Raises:
        ValueError: If the specified color space is not supported.
    """

    conversion_codes = {
        "HSV": cv2.COLOR_BGR2HSV,
        "LAB": cv2.COLOR_BGR2LAB,
        "HSL": cv2.COLOR_BGR2HLS,
    }

    if color_space not in conversion_codes:
        raise ValueError(f"Unsupported color space: {color_space}")

    return cv2.cvtColor(img, conversion_codes[color_space])


def orb_feature_detector(image: np.ndarray) -> np.ndarray:
    """
    This function detects features in an image using the ORB algorithm.

    Args:
        image: A numpy array representing the input image in BGR color space.

    Returns:
        A numpy array representing the grayscale image with detected ORB features visualized as circles.

    Raises:
        TypeError: If the input image is not a numpy array.
    """

    # Check if the input is a numpy array
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy array")

    # Convert the image to grayscale as ORB works best with grayscale images
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create an ORB detector with 1500 feature points (adjust this value as needed)
    orb = cv2.ORB_create(nfeatures=1500)

    # Detect keypoints and compute descriptors using ORB
    keypoints_orb, descriptors = orb.detectAndCompute(grayscale_image, None)

    # Draw the detected keypoints on the grayscale image for visualization
    image_with_detections = cv2.drawKeypoints(grayscale_image, keypoints_orb, None)

    return image_with_detections


def match_key_points_between_two_images(
    image1: np.ndarray, image2: np.ndarray
) -> np.ndarray:
    """
    This function matches keypoints between two images using the ORB detector
    and the Brute-Force Matcher with Hamming distance.

    Args:
        image1: A numpy array representing the first image in BGR color space.
        image2: A numpy array representing the second image in BGR color space.

    Returns:
        A numpy array representing the grayscale image from the first image
        with matched keypoints visualized as lines connecting corresponding points
        in both images.

    Raises:
        TypeError: If any of the input images are not numpy arrays.
    """

    # Load the images
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Create the ORB detector
    orb = cv2.ORB_create(nfeatures=1500)

    # Detect and compute the keypoints and descriptors for both images
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Create the descriptor matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Find the matches
    matches = bf.match(descriptors1, descriptors2)

    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the matches
    img_matches = cv2.drawMatches(
        img1, keypoints1, img2, keypoints2, matches, None, flags=2
    )

    return img_matches


def cartoonization(image, sigma_s=130, sigma_r=0.07):
    """
    Applies cartoonization effect to an image using bilateral filtering.

    Args:
        image (numpy.ndarray): A NumPy array representing the image in BGR
                               color format.
        sigma_s (float, optional): Controls the filtering strength for
                                   preserving edges. Defaults to 130.
        sigma_r (float, optional): Controls the influence of nearby pixels.
                                   Defaults to 0.07.

    Returns:
        numpy.ndarray: A NumPy array representing the cartoonized image.
    """
    cartoonized = cv2.stylization(image, sigma_s, sigma_r)
    return cartoonized
