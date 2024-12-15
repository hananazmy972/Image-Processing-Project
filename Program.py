import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
import numpy as np
from PIL import Image as PILImage, ImageTk
import matplotlib.pyplot as plt

# Class to represent an image
class Image:
    def __init__(self):
        self.image_data = None

    def load_image(self, file_path):
        self.image_data = cv2.imread(file_path)

    def save_image(self, file_path):
        if self.image_data is not None:
            cv2.imwrite(file_path, self.image_data)

    def convert_to_grayscale(self):
        if self.image_data is not None:
            self.image_data = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2GRAY)
    

# Class for image processing algorithms
class ImageProcessor:
        ## CANNY Steps ## 
    def gaussian_kernel(size, sigma=1):
        """Generate a Gaussian kernel."""
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
            (size, size)
        )
        return kernel / np.sum(kernel)

    def sobel_filters(img):
        """Apply Sobel filters to compute gradients."""
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        Gx = cv2.filter2D(img, cv2.CV_64F, Kx)
        Gy = cv2.filter2D(img, cv2.CV_64F, Ky)

        magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
        angle = np.arctan2(Gy, Gx)
        return magnitude, angle

    def non_maximum_suppression(magnitude, angle):
        """Suppress non-maximum pixels in the gradient direction."""
        H, W = magnitude.shape
        output = np.zeros((H, W), dtype=np.float32)
        angle = angle * 180.0 / np.pi
        angle[angle < 0] += 180

        for i in range(1, H - 1):
            for j in range(1, W - 1):
                q = 255
                r = 255

                # Check the gradient direction
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                # Suppress if not a local maximum
                if magnitude[i, j] >= q and magnitude[i, j] >= r:
                    output[i, j] = magnitude[i, j]

        return output

    def double_threshold(img, low_threshold, high_threshold):
        """Apply double thresholding."""
        strong = 255
        weak = 50

        strong_i, strong_j = np.where(img >= high_threshold)
        weak_i, weak_j = np.where((img >= low_threshold) & (img < high_threshold))

        output = np.zeros_like(img, dtype=np.uint8)
        output[strong_i, strong_j] = strong
        output[weak_i, weak_j] = weak

        return output, weak, strong

    def edge_tracking_by_hysteresis(img, weak, strong):
        """Perform edge tracking by hysteresis."""
        H, W = img.shape
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if img[i, j] == weak:
                    # Check if any neighbor is strong
                    if (img[i + 1, j - 1:j + 2] == strong).any() or (img[i - 1, j - 1:j + 2] == strong).any() or (img[i, [j - 1, j + 1]] == strong).any():
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
        return img
    @staticmethod
    def canny_edge_detection(image, low_threshold=50, high_threshold=150):
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply Gaussian blur
        kernel = ImageProcessor.gaussian_kernel(size=5, sigma=1)
        smoothed = cv2.filter2D(gray, cv2.CV_64F, kernel)

        # Compute gradient magnitude and direction
        magnitude, angle = ImageProcessor.sobel_filters(smoothed)

        # Apply non-maximum suppression
        suppressed = ImageProcessor.non_maximum_suppression(magnitude, angle)

        # Apply double threshold
        thresholded, weak, strong = ImageProcessor.double_threshold(suppressed, low_threshold, high_threshold)

        # Perform edge tracking by hysteresis
        edges = ImageProcessor.edge_tracking_by_hysteresis(thresholded, weak, strong)

        return edges

    @staticmethod
    def sobel_edge_detection(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        return np.uint8(np.clip(sobel, 0, 255))

    @staticmethod
    def prewitt_edge_detection(image):
            # Custom grayscale conversion
            def to_grayscale(image):
                gray_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        r, g, b = image[i, j]
                        gray = int(0.299 * r + 0.587 * g + 0.114 * b)
                        gray_image[i, j] = gray
                return gray_image

            gray = to_grayscale(image)  # Convert to grayscale 
            img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)  # Reduce noise

            # Define Prewitt Kernels
            kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])  # Detects horizontal edges
            kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # Detects vertical edges

            # Apply the Kernels
            img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
            img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)

            return img_prewittx + img_prewitty

    @staticmethod
    def difference_of_gaussian(image):
        # Parameters for Gaussian blur
        small_kernel = (3, 3)
        large_kernel = (11, 11)
        sigma = 0  # Auto-calculated by OpenCV based on kernel size

        # Check if the image is colorful (3 channels)
        if len(image.shape) == 3 and image.shape[2] == 3:  # Colorful image
            # Split the image into its color channels (R, G, B)
            channels = cv2.split(image)
            processed_channels = []

            # Apply DoG on each channel
            for channel in channels:
                # Gaussian Blur with Smaller Kernel
                blur1 = cv2.GaussianBlur(channel, small_kernel, sigma)
                # Gaussian Blur with Larger Kernel
                blur2 = cv2.GaussianBlur(channel, large_kernel, sigma)
                # Subtract the Larger Blur from the Smaller Blur and amplify edges
                dog_channel = cv2.subtract(blur1, blur2)
                dog_channel = cv2.convertScaleAbs(dog_channel, alpha=2.0, beta=0)  # Scale to enhance edges
                processed_channels.append(dog_channel)

            # Merge processed channels back into a single image
            dog = cv2.merge(processed_channels)
        else:  # Grayscale image
            # Gaussian Blur with Smaller Kernel
            blur1 = cv2.GaussianBlur(image, small_kernel, sigma)
            # Gaussian Blur with Larger Kernel
            blur2 = cv2.GaussianBlur(image, large_kernel, sigma)
            # Subtract the Larger Blur from the Smaller Blur and amplify edges
            dog = cv2.subtract(blur1, blur2)
            dog = cv2.convertScaleAbs(dog, alpha=2.0, beta=0)  # Scale to enhance edges

        return dog

    @staticmethod
    def add_noise(image, noise_level):
        # Generate noise
        noise = np.random.normal(0, noise_level, image.shape).astype(np.uint8)
        
        # Convert image and noise to int16 for addition to avoid overflow
        image_int16 = image.astype(np.int16)
        noise_int16 = noise.astype(np.int16)
        
        # Add noise to the image
        noisy_image = image_int16 + noise_int16
        
        # Manually clip values to ensure they are in the range [0, 255]
        rows, cols = noisy_image.shape[:2]
        if len(noisy_image.shape) == 3:  # For color images
            channels = noisy_image.shape[2]
            for row in range(rows):
                for col in range(cols):
                    for channel in range(channels):
                        noisy_image[row, col, channel] = max(0, min(255, noisy_image[row, col, channel]))
        else:  # For grayscale images
            for row in range(rows):
                for col in range(cols):
                    noisy_image[row, col] = max(0, min(255, noisy_image[row, col]))
        
        # Convert back to uint8
        return noisy_image.astype(np.uint8)

    @staticmethod
    def remove_noise(image):
        # Define the Gaussian kernel
        kernel_size = 5
        sigma = 1.0
        kernel = ImageProcessor.gaussian_kernel(kernel_size, sigma)

        # Check if the image is colorful
        if len(image.shape) == 3:  # Colorful image (3 channels)
            # Split the image into channels
            channels = cv2.split(image)
            processed_channels = []

            for channel in channels:
                # Get image dimensions
                height, width = channel.shape
                padding = kernel_size // 2

                # Pad the channel to handle border pixels
                padded_channel = np.pad(channel, padding, mode='constant', constant_values=0)
                result = np.zeros_like(channel, dtype=np.uint8)

                # Apply the Gaussian kernel to each pixel
                for i in range(height):
                    for j in range(width):
                        region = padded_channel[i:i + kernel_size, j:j + kernel_size]
                        result[i, j] = np.sum(region * kernel)

                processed_channels.append(result)

            # Merge the processed channels back into a single image
            return cv2.merge(processed_channels)

        else:  # Grayscale image (single channel)
            # Get image dimensions
            height, width = image.shape
            padding = kernel_size // 2

            # Pad the image to handle border pixels
            padded_image = np.pad(image, padding, mode='constant', constant_values=0)
            result = np.zeros_like(image, dtype=np.uint8)

            # Apply the Gaussian kernel to each pixel
            for i in range(height):
                for j in range(width):
                    region = padded_image[i:i + kernel_size, j:j + kernel_size]
                    result[i, j] = np.sum(region * kernel)

            return result


    @staticmethod
    def histogram_equalization(image):
        # Step 1: Convert to Grayscale
        rows, cols, _ = image.shape
        gray = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                r, g, b = image[i, j]
                gray[i, j] = int(0.299 * r + 0.587 * g + 0.114 * b)
 
        # Step 2: Calculate Histogram
        hist = np.zeros(256, dtype=int)
        for i in range(rows):
            for j in range(cols):
                hist[gray[i, j]] += 1
 
        # Step 3: Calculate CDF Manually
        cdf = np.zeros(256, dtype=int)
        cdf[0] = hist[0]
        for i in range(1, 256):
            cdf[i] = cdf[i - 1] + hist[i]
 
        # Step 4: Normalize the CDF
        cdf_min = next(c for c in cdf if c > 0)  # First non-zero value in the CDF
        total_pixels = rows * cols
        cdf_normalized = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            cdf_normalized[i] = int(((cdf[i] - cdf_min) / (total_pixels - cdf_min)) * 255)
 
        # Step 5: Map Original Pixels to Equalized Values
        equalized = np.zeros_like(gray, dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                equalized[i, j] = cdf_normalized[gray[i, j]]
 
        # Step 6: Convert Back to 3-Channel Image
        equalized_bgr = np.stack([equalized] * 3, axis=-1)
 
        return equalized_bgr

    @staticmethod
    def show_histogram(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plt.hist(gray.ravel(), bins=256, range=(0, 256))
        plt.title("Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.show()

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Segmentation Tool")
        self.root.config(bg="#D4BDAC")
        # Set window size
        self.root.geometry("1000x800")

        self.image = Image()
        self.processed_image = None

        # Load and set background image
        bg_image = PILImage.open("background.jpg")  # You'll need to provide this image
        bg_image = bg_image.resize((1000, 800))
        self.bg_photo = ImageTk.PhotoImage(bg_image)
        
        # Create background label
        bg_label = tk.Label(root, image=self.bg_photo)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Create frames for better organization
        self.button_frame = tk.Frame(root, bg="#ebddcc")
        self.button_frame.pack(side=tk.LEFT, padx=100, pady=20)

        self.canvas = tk.Label(root, bg="#ebddcc")
        self.canvas.place(x=434, y=128) 

        # Create buttons with rounded edges
        button_style = {'bg': "#be9764", 'fg': "white", 'width': 20 ,'font': ('Arial', 13) }

        # Create buttons
        self.load_button = tk.Button(self.button_frame, text="Load Image", command=self.load_image,**button_style)
        self.show_original_button = tk.Button(self.button_frame, text="Show Original Image", command=self.show_original,**button_style)
        self.canny_button = tk.Button(self.button_frame, text="Canny Edge Detection", command=self.apply_canny,**button_style)
        self.sobel_button = tk.Button(self.button_frame, text="Sobel Edge Detection", command=self.apply_sobel,**button_style)
        self.prewitt_button = tk.Button(self.button_frame, text="Prewitt Edge Detection", command=self.apply_prewitt,**button_style)
        self.dog_button = tk.Button(self.button_frame, text="Difference of Gaussian", command=self.apply_dog,**button_style)
        self.add_noise_button = tk.Button(self.button_frame, text="Add Noise", command=self.apply_add_noise,**button_style)
        self.remove_noise_button = tk.Button(self.button_frame, text="Remove Noise", command=self.apply_remove_noise,**button_style)
        self.hist_eq_button = tk.Button(self.button_frame, text="Histogram Equalization", command=self.apply_histogram_equalization,**button_style)
        self.histogram_button = tk.Button(self.button_frame, text="Show Histogram", command=self.show_histogram, **button_style)
        self.save_button = tk.Button(self.button_frame, text="Save Processed Image", command=self.save_image,**button_style)

        # Noise level slider
        self.noise_level = tk.IntVar(value=5)
        self.noise_slider = tk.Scale(self.button_frame, from_=0, to=10, orient="horizontal", label="Noise Level", variable=self.noise_level ,bg="#ebddcc", length=200,width=20,font=("Arial", 12))

        # Pack buttons and slider
        buttons = [
            self.load_button, self.show_original_button, self.canny_button,
            self.sobel_button, self.prewitt_button, self.dog_button,
            self.add_noise_button, self.remove_noise_button,
            self.hist_eq_button, self.histogram_button, self.save_button
        ]

        for button in buttons:
            button.pack(pady=9)

        self.noise_slider.pack(pady=10)

    def display_image(self, img_data):
        img_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        img_pil = PILImage.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
         # Resize the PIL image before converting it to PhotoImage
        img_pil = img_pil.resize((488, 528))  # Adjust to desired size or use dynamic scaling

        # Convert to PhotoImage
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.config(image=img_tk)
        self.canvas.image = img_tk

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.bmp")])
        if file_path:
            self.image.load_image(file_path)
            self.processed_image = None
            self.display_image(self.image.image_data)

    def show_original(self):
        if self.image.image_data is not None:
            self.display_image(self.image.image_data)

    def apply_canny(self):
        if self.image.image_data is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        
        # Ask for low and high thresholds
        low_threshold = 50  # Set default or dynamically ask the user for input
        high_threshold = 150  # Set default or dynamically ask the user for input
        
        # Apply Canny edge detection
        self.processed_image = ImageProcessor.canny_edge_detection(self.image.image_data, low_threshold, high_threshold)
        
        # Display the processed image (grayscale converted to RGB for display)
        self.display_image(cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR))

    def apply_sobel(self):
        self.process_and_display(ImageProcessor.sobel_edge_detection)

    def apply_prewitt(self):
        self.process_and_display(ImageProcessor.prewitt_edge_detection)

    def apply_dog(self):
        self.process_and_display(ImageProcessor.difference_of_gaussian)

    def apply_add_noise(self):
        self.process_and_display(lambda img: ImageProcessor.add_noise(img, self.noise_level.get()))

    def apply_remove_noise(self):
        self.process_and_display(ImageProcessor.remove_noise)

    def apply_histogram_equalization(self):
        self.process_and_display(ImageProcessor.histogram_equalization)

    def show_histogram(self):
        if self.image.image_data is not None:
            ImageProcessor.show_histogram(self.image.image_data)

    def process_and_display(self, process_function):
        if self.image.image_data is None:
            messagebox.showerror("Error", "Load an image first!")
            return
        self.processed_image = process_function(self.image.image_data)

        # Handle grayscale and color images separately
        if len(self.processed_image.shape) == 2:  # Grayscale image
            self.display_image(cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR))
        else:  # Color image
            self.display_image(self.processed_image)


    def save_image(self):
        if self.processed_image is None:
            messagebox.showerror("Error", "No processed image to save!")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
        if file_path:
            cv2.imwrite(file_path, self.processed_image)
            messagebox.showinfo("Save Image", "Image saved successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()