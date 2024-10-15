from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000


def resize_tiff(input_path, output_path, size=(1024, 1024)):
    # Open the TIFF image
    with Image.open(input_path) as img:
        # Resize the image while maintaining aspect ratio
        img.thumbnail(size)
        
        # Save the resized image
        img.save(output_path, format="TIFF")
        print(f"Image saved to {output_path}")

# Example usage:
input_tiff = r"D:\Subhash Drone Path\yolov5\OriginalImages\yp_upload1.tif"  # Replace with your file path
output_tiff = "resized_image3.png"  # Replace with desired output file path
resize_tiff(input_tiff, output_tiff, size=(1024, 1024))  # Resize to 1024x1024
