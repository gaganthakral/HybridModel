import matplotlib.pyplot as plt
import cv2
from skimage import io
from medpy.filter.smoothing import anisotropic_diffusion
from skimage import img_as_float
from skimage.metrics import peak_signal_noise_ratio


noisy_img = img_as_float(io.imread("./Img/LDCT_Noisy.jpeg", as_gray=True))
ref_img = img_as_float(io.imread("./Img/LDCT_Clean.jpeg"))

img_aniso_filtered = anisotropic_diffusion(noisy_img, niter=50, kappa=50, gamma=0.2, option=2)

noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
anisotropic_cleaned_psnr = peak_signal_noise_ratio(ref_img, img_aniso_filtered)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", anisotropic_cleaned_psnr)


plt.imshow(img_aniso_filtered, cmap='gray')
plt.imsave("./Img/anisotropic_denoised.jpeg", img_aniso_filtered, cmap='gray')
