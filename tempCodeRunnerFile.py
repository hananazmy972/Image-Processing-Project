    @staticmethod
    def add_noise(image, noise_level):
        """Add Gaussian noise to an image with reduced reliance on built-in methods."""
        if len(image.shape) == 2:  # Grayscale image
            noisy_image = np.zeros_like(image, dtype=np.int16)  # Intermediate result to prevent overflow
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    noise = int(noise_level * (np.random.rand() - 0.5) * 2)  # Generate noise in range [-noise_level, noise_level]
                    noisy_image[i, j] = image[i, j] + noise
        else:  # Color image
            noisy_image = np.zeros_like(image, dtype=np.int16)  # Intermediate result to prevent overflow
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    for k in range(image.shape[2]):  # For each color channel
                        noise = int(noise_level * (np.random.rand() - 0.5) * 2)  # Generate noise
                        noisy_image[i, j, k] = image[i, j, k] + noise

        # Clip the noisy image to valid pixel range [0, 255]
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image.astype(np.uint8)