import cv2


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
