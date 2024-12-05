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
    @staticmethod
    def canny_edge_detection(image):
        return cv2.Canny(image, 100, 200)

    @staticmethod
    def sobel_edge_detection(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        return np.uint8(np.clip(sobel, 0, 255))

    @staticmethod
    def prewitt_edge_detection(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewittx = cv2.filter2D(gray, -1, kernelx)
        prewitty = cv2.filter2D(gray, -1, kernely)
        return prewittx + prewitty

    @staticmethod
    def difference_of_gaussian(image):
        blur1 = cv2.GaussianBlur(image, (5, 5), 0)
        blur2 = cv2.GaussianBlur(image, (9, 9), 0)
        return cv2.subtract(blur1, blur2)

    @staticmethod
    def add_noise(image, noise_level):
        if len(image.shape) == 2:  # Grayscale image
            noise = np.random.normal(0, noise_level, image.shape).astype(np.uint8)
        else:  # Color image
            noise = np.random.normal(0, noise_level, image.shape).astype(np.uint8)
        return cv2.add(image, noise)

    
    @staticmethod
    def remove_noise(image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    @staticmethod
    def histogram_equalization(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def show_histogram(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plt.hist(gray.ravel(), bins=256, range=(0, 256))
        plt.title("Histogram")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.show()

# GUI Class
class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Segmentation Tool")
        self.image = Image()
        self.processed_image = None

        # Noise level slider
        self.noise_level = tk.IntVar(value=5)

        # UI Components
        self.canvas = tk.Label(root)
        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.show_original_button = tk.Button(root, text="Show Original Image", command=self.show_original)
        self.canny_button = tk.Button(root, text="Canny Edge Detection", command=self.apply_canny)
        self.sobel_button = tk.Button(root, text="Sobel Edge Detection", command=self.apply_sobel)
        self.prewitt_button = tk.Button(root, text="Prewitt Edge Detection", command=self.apply_prewitt)
        self.dog_button = tk.Button(root, text="Difference of Gaussian", command=self.apply_dog)
        self.add_noise_button = tk.Button(root, text="Add Noise", command=self.apply_add_noise)
        self.remove_noise_button = tk.Button(root, text="Remove Noise", command=self.apply_remove_noise)
        self.hist_eq_button = tk.Button(root, text="Histogram Equalization", command=self.apply_histogram_equalization)
        self.histogram_button = tk.Button(root, text="Show Histogram", command=self.show_histogram)
        self.save_button = tk.Button(root, text="Save Processed Image", command=self.save_image)
        self.grayscale_button = tk.Button(root, text="Convert to Grayscale", command=self.convert_to_grayscale)

        self.noise_slider = tk.Scale(root, from_=0, to=10, orient="horizontal", label="Noise Level",
                                     variable=self.noise_level)

        # Layout
        self.load_button.grid(row=0, column=0, padx=10, pady=10)
        self.show_original_button.grid(row=1, column=0, padx=10, pady=10)
        self.canny_button.grid(row=2, column=0, padx=10, pady=10)
        self.sobel_button.grid(row=3, column=0, padx=10, pady=10)
        self.prewitt_button.grid(row=4, column=0, padx=10, pady=10)
        self.dog_button.grid(row=5, column=0, padx=10, pady=10)
        self.add_noise_button.grid(row=6, column=0, padx=10, pady=10)
        self.remove_noise_button.grid(row=7, column=0, padx=10, pady=10)
        self.hist_eq_button.grid(row=8, column=0, padx=10, pady=10)
        self.histogram_button.grid(row=9, column=0, padx=10, pady=10)
        self.save_button.grid(row=10, column=0, padx=10, pady=10)
        self.grayscale_button.grid(row=11, column=0, padx=10, pady=10)
        self.noise_slider.grid(row=12, column=0, padx=10, pady=10)
        self.canvas.grid(row=0, column=1, rowspan=13, padx=10, pady=10)

    def display_image(self, img_data):
        img_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        img_pil = PILImage.fromarray(img_rgb)
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
        self.process_and_display(ImageProcessor.canny_edge_detection)

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

    def convert_to_grayscale(self):
        if self.image.image_data is not None:
            self.image.convert_to_grayscale()
            self.display_image(cv2.cvtColor(self.image.image_data, cv2.COLOR_GRAY2BGR))

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