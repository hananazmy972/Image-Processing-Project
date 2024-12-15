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