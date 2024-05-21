import cv2

# Load an image
img = cv2.imread("data/orange_frog.jpg")

# Check if image loaded successfully
if img is None:
    print("Error: Could not read image!")
    exit(1)

# Get the original image dimensions
orig_height, orig_width = img.shape[:2]

# Define the new desired width (e.g., 300 pixels)
decrease_rate = 2

# Resize the image
resized_img = cv2.resize(
    img, (int(orig_width / decrease_rate), int(orig_height / decrease_rate))
)

# Save the resized image (replace "resized_image.jpg" with your desired filename)
cv2.imwrite("data/resized_image.jpg", resized_img)
print("Image saved as resized_image.jpg")
