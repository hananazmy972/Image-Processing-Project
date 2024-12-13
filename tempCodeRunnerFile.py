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
