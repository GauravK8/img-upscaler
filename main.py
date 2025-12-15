import cv2
import numpy as np
from PIL import Image

def sharpen_image_opencv(image_path, output_path):
    """Sharpens an image using a common sharpening kernel with OpenCV."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return

    # Define a sharpening kernel
    sharpen_kernel = np.array([[-1,-1,-1], 
                               [-1, 9,-1], 
                               [-1,-1,-1]])
    
    # Apply the sharpening kernel
    sharpened_image = cv2.filter2D(image, -1, sharpen_kernel)
    cv2.imwrite(output_path, sharpened_image)
    print(f"Image sharpened and saved to {output_path}")

def upscale_image_pillow(image_path, scale_factor, output_path):
    """Upscales an image using Pillow with a high-quality resampling filter."""
    original_image = Image.open(image_path)
    new_width = int(original_image.width * scale_factor)
    new_height = int(original_image.height * scale_factor)
    
    # Use Image.LANCZOS for high-quality upscaling
    upscaled_image = original_image.resize((new_width, new_height), Image.LANCZOS)
    upscaled_image.save(output_path)
    print(f"Image upscaled and saved to {output_path}")

# Example usage:
# upscale_image_pillow('input.jpg', 2, 'upscaled_output.jpg') 

# Load the image
image = cv2.imread('download.png')

# Define the desired scaling factor (e.g., 2 for 2x upscale)
scale_factor = 3

# Calculate the new dimensions
new_width = int(image.shape[1] * scale_factor)
new_height = int(image.shape[0] * scale_factor)

# Upscale the image using bicubic interpolation
upscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

# Save the upscaled image
cv2.imwrite('upscaled_image_bicubic.jpg', upscaled_image)

sharpen_image_opencv('upscaled_image_bicubic.jpg', 'sharpened_output.jpg')
upscale_image_pillow('upscaled_image_bicubic.jpg',1,'sharpened_output_upscaled.jpg')

print("Image upscaled using Bicubic interpolation.")

