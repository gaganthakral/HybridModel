import numpy as np
import cv2

# Load LDCT image
ldct_img = cv2.imread("./Img/LDCT_Clean.jpeg", cv2.IMREAD_GRAYSCALE)

# Add Gaussian noise
mean = 0
variance = 0.1
sigma = np.sqrt(variance)
gaussian_noise = np.random.normal(mean, sigma, ldct_img.shape)
ldct_img_with_noise = ldct_img + gaussian_noise

# Add Salt-and-Pepper noise
noise_amount = 0.05
salt_pepper_noise = np.zeros(ldct_img.shape, dtype=np.uint8)
salt_amount = int(noise_amount * ldct_img.size * 0.5)
salt_coords = [np.random.randint(0, i - 1, salt_amount) for i in ldct_img.shape]
salt_pepper_noise[tuple(salt_coords)] = 255
pepper_amount = int(noise_amount * ldct_img.size * 0.5)
pepper_coords = [np.random.randint(0, i - 1, pepper_amount) for i in ldct_img.shape]
salt_pepper_noise[tuple(pepper_coords)] = 0
ldct_img_with_noise = cv2.add(ldct_img, salt_pepper_noise)

cv2.imwrite("./Img/LDCT_Noisy.jpeg", ldct_img_with_noise)
# Display original and noisy images
cv2.imshow("LDCT Image", ldct_img)
cv2.imshow("LDCT Image with Gaussian Noise", ldct_img_with_noise)
cv2.waitKey(0)
cv2.destroyAllWindows()

