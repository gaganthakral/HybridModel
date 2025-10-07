from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import img_as_float
from skimage import io
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt

noisy_img = img_as_float(io.imread("./Img/LDCT_Noisy.jpeg"))
ref_img = img_as_float(io.imread("./Img/LDCT_Clean.jpeg"))
denoise_bilateral = denoise_bilateral(noisy_img, sigma_spatial=10, channel_axis=-1)
noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
bilateral_cleaned_psnr = peak_signal_noise_ratio(ref_img, denoise_bilateral)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", bilateral_cleaned_psnr)
plt.imsave("./Img/bilateral_smoothed.jpeg", denoise_bilateral, cmap='gray')
