# Image Segmentation and Processing Toolkit

## Overview

This project is a comprehensive **Image Segmentation and Processing Toolkit** designed to process, enhance, and segment images using various edge detection and filtering algorithms. The application uses **Tkinter** for the user interface, **OpenCV** for image manipulation, and **NumPy** for numerical computations. The project demonstrates the implementation of classical computer vision techniques for segmentation, noise handling, and histogram equalization.
<img src="https://drive.google.com/file/d/1OqBkZBmujY4S21V771tmotHT-JYS4wHN/view?usp=sharing" width="1000">

## Features

- **Image Loading and Saving**: Load images from local storage and save processed results.
- **Grayscale Conversion**: Convert colored images into grayscale for further processing.
- **Edge Detection Algorithms**:
  - Canny Edge Detection
  - Sobel Edge Detection
  - Prewitt Edge Detection
- **Difference of Gaussian (DoG)**: Detect edges by subtracting two Gaussian-blurred versions of the image.
- **Noise Addition and Removal**:
  - Add Gaussian noise to simulate real-world conditions.
  - Remove noise using Gaussian blurring.
- **Histogram Equalization**: Enhance image contrast by redistributing pixel intensity.
- **Show Histogram**: Generate and display histograms to analyze the distribution of pixel intensity values in images.

## Algorithms Description

### 1. **Canny Edge Detection**
Canny is a multi-stage edge detection algorithm that involves:
- **Gaussian Smoothing**: Reduces noise by applying a Gaussian filter.
- **Gradient Computation**: Uses Sobel filters to compute intensity gradients in the x and y directions.
- **Non-Maximum Suppression**: Removes pixels that are not local maxima along the gradient direction.
- **Double Thresholding**: Identifies strong and weak edges based on intensity thresholds.
- **Edge Tracking by Hysteresis**: Connects weak edges to strong edges if they are part of a continuous boundary.

**Usage**: Best for detecting clean, precise edges in noisy images.

### 2. **Sobel Edge Detection**
The Sobel method calculates the gradient of image intensity:
- Applies kernels to detect horizontal and vertical edges.
- Combines the results to calculate the magnitude of the gradient.

**Usage**: Effective for detecting edges where noise is minimal.

### 3. **Prewitt Edge Detection**
Similar to Sobel, Prewitt applies convolution kernels for horizontal and vertical edge detection:
- Horizontal kernel emphasizes changes in the x-direction.
- Vertical kernel emphasizes changes in the y-direction.

**Usage**: Simpler and faster but less accurate than Sobel.

### 4. **Difference of Gaussian (DoG)**
DoG enhances edges by subtracting two Gaussian-blurred images with different standard deviations. This highlights regions with significant intensity differences.

**Usage**: Ideal for blob detection and edge enhancement.

### 5. **Noise Handling**
- **Add Noise**: Simulates real-world conditions by adding Gaussian noise.
- **Remove Noise**: Reduces noise using a Gaussian blur, smoothing high-frequency variations.

**Usage**: Useful for testing the robustness of algorithms against noisy data.

### 6. **Histogram Equalization**
Improves contrast by redistributing pixel intensities based on the cumulative distribution function (CDF). This ensures a uniform distribution of intensity levels.

**Usage**: Enhances images with poor lighting conditions or low contrast.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- PIL (Pillow)
- Matplotlib (for optional visualizations)

## Usage Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/image-segmentation-toolkit.git
   cd image-segmentation-toolkit
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python main.py
   ```

4. **Load an Image**: Use the file dialog to select an image.
5. **Apply Processing**: Select the desired algorithm (Canny, Sobel, etc.) and view the results.
6. **Save Results**: Save the processed image back to local storage.

## Contributions
Feel free to contribute by adding new features or optimizing existing algorithms. Submit a pull request with detailed comments.

## License
This project is licensed under the MIT License.

