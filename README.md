# Image Segmentation Tool

This project is a Python-based image processing tool built using **Tkinter** for the graphical user interface (GUI), **OpenCV** for image processing, and **PIL** for displaying images. The tool allows users to load images, apply various image processing algorithms, and save the processed images.

## Features

- **Load and Display Images**: Load images from local storage and display them in the GUI.
- **Edge Detection**: Apply different edge detection algorithms such as Canny, Sobel, and Prewitt.
- **Noise Addition and Removal**: Add random noise to images and remove it using Gaussian blur.
- **Histogram Equalization**: Improve the contrast of grayscale images.
- **Grayscale Conversion**: Convert color images to grayscale.
- **Histogram Visualization**: Show the pixel intensity histogram of the image.

## Image Processing Algorithms

The following image processing techniques are implemented in this project:

1. **Canny Edge Detection**: Detect edges in the image using the Canny edge detection algorithm.
2. **Sobel Edge Detection**: Apply Sobel filter for edge detection in horizontal and vertical directions.
3. **Prewitt Edge Detection**: Use Prewitt operator for edge detection.
4. **Difference of Gaussian**: Perform edge detection using a difference of Gaussian filter.
5. **Add Noise**: Add Gaussian noise to the image.
6. **Remove Noise**: Remove noise by applying Gaussian blur.
7. **Histogram Equalization**: Equalize the histogram of a grayscale image to enhance contrast.

## Requirements

- Python 3.x
- Tkinter
- OpenCV
- NumPy
- PIL (Pillow)
- Matplotlib


##Usage

1. Load an image: Click on the "Load Image" button to select an image from your local storage.
2. Apply image processing: Use the buttons on the left to apply various image processing algorithms like edge detection, noise addition/removal, or grayscale conversion.
3. View the result: The processed image will be displayed on the right canvas.
4. Save the processed image: Once the processing is done, click the "Save Processed Image" button to save the image to your desired location.
