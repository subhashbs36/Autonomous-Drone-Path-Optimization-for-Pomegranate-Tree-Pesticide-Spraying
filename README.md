
# Autonomous Drone Path Optimization for Pomegranate Tree Pesticide Spraying

This project is designed to autonomously determine the most efficient path for a drone to spray pesticides over individual pomegranate trees in a large farm. Using the Travelling Salesman Problem (TSP) algorithm, YOLOv5 for crop detection, and advanced image processing techniques, the tool calculates the optimal GPS-based path for the drone, ensuring efficient pesticide spraying.

**Original Image:**
![Original Image](Screenshots/Original.png)

**Bounding Box Image:**
![Bounding Box Image](Screenshots/bbox_detection.png)

**Drone Path Image 1:**
![Drone Path Image 1](Screenshots/resized_image1.png)

**Drone Path Image 2:**
![Drone Path Image 2](Screenshots/resized_image2.png)

## Features

- **TIFF Image Processing:** Input a high-resolution TIFF image of a pomegranate farm with GPS coordinates for precise tree detection.
- **Image Resizing and Slicing:** Resizes the image for efficient processing without losing critical data, followed by image slicing for quicker analysis.
- **YOLOv5 Crop Detection:** Runs YOLOv5 on each sliced image to detect individual pomegranate trees.
- **BBox Center Calculation:** Calculates and stores the center of the bounding boxes for detected trees to extract GPS coordinates.
- **Efficient Path Calculation:** Utilizes the Travelling Salesman Problem (TSP) algorithm to compute the most efficient path for the drone to spray pesticides.
- **GPS Path Extraction:** Extracts the GPS coordinates of the tree centers in the optimal path, enabling the drone to hover precisely over each tree.
- **Visual Path Display:** Displays the optimal drone path on the image, from the starting to the ending coordinates, input by the user.

## Methodology

1. **TIFF Image Input:**
   - Load a TIFF image of the pomegranate farm, containing over 600 trees, along with associated GPS coordinates.

2. **Image Resizing and Slicing:**
   - Resize the image for better processing efficiency without losing any important data.
   - Slice the image into smaller segments to allow faster processing through the YOLOv5 model.

3. **Crop Detection with YOLOv5:**
   - Detect pomegranate trees in each sliced image using the YOLOv5 model.
   - Overlap correction is applied to ensure accuracy when merging results from different slices.

4. **Bounding Box Center Calculation:**
   - For each detected pomegranate tree, calculate the center of its bounding box.
   - Store the calculated centers to extract the GPS coordinates for each tree.

5. **TSP Path Optimization:**
   - With all tree centers identified, use the TSP algorithm to calculate the most efficient spraying path.
   - The algorithm processes around 600+ trees and takes approximately 20 minutes to compute the optimal path.

6. **GPS Path Extraction and Visualization:**
   - After determining the most efficient path, extract the GPS coordinates for each tree.
   - Visually display the optimal path on the farm's image, starting and ending at user-specified coordinates.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/drone-path-optimization.git
   ```

2. Navigate to the project directory:
   ```bash
   cd drone-path-optimization
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

- Python 3.8+
- YOLOv5
- OpenCV
- NumPy
- Matplotlib
- TSP solver library
- GDAL (for handling geospatial TIFF images)

To install all dependencies, use:
```bash
pip install opencv-python numpy matplotlib tsp_solver gdal
```

## Usage

1. Launch the tool:
   ```bash
   python main.py
   ```

2. Load the TIFF image of the pomegranate farm.
   
3. Input the starting and ending GPS coordinates for the drone's path.

4. The tool will perform the following:
   - Resize and slice the image.
   - Detect pomegranate trees using YOLOv5.
   - Calculate the center of each tree and extract GPS coordinates.
   - Run the TSP algorithm to compute the optimal path.
   - Display the calculated path visually on the image.
   
5. Finally, the GPS coordinates of the path are extracted for drone navigation.

## Example

Here’s an example of the visual path output for a pomegranate farm:

![Example Path](path/to/example-image.png)

## Applications

- **Agricultural Automation:** Optimize pesticide spraying in pomegranate or other crop farms using autonomous drones.
- **Precision Farming:** Reduce resource waste and increase efficiency by ensuring precise spraying locations.
- **Scalable Drone Operations:** Easily adapt this approach to larger farms with different crop layouts.

## Reference

This method follows the approach outlined in the paper:  
**"Efficient Patch-Wise Crop Detection Algorithm for UAV-Generated Orthomosaic"**  
Published in Springer. [Read here](https://doi.org/10.1007/978-981-99-8684-2_14)

## Future Enhancements

- Implementing real-time path recalibration during drone flight.
- Enhancing overlap correction for more accurate detection.
- Extending the approach to other crops with different farm structures.

## Contributing

Contributions are welcome! Feel free to submit issues, fork the repository, and send pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
