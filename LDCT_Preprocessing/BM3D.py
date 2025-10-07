import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio
import bm3d
import numpy as np

noisy_img = img_as_float(io.imread("./Img/LDCT_Noisy.jpeg"))
ref_img = img_as_float(io.imread("./Img/LDCT_Clean.jpeg"))

BM3D_denoised_image = bm3d.bm3d(noisy_img, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.ALL_STAGES)
noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
BM3D_cleaned_psnr = peak_signal_noise_ratio(ref_img, BM3D_denoised_image)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", BM3D_cleaned_psnr)


plt.imshow(BM3D_denoised_image, cmap='gray')
plt.show()
plt.imsave("./Img/BM3D_denoised.jpeg", BM3D_denoised_image, cmap='gray')
