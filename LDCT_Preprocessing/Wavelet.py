from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import img_as_float
from skimage import io
from skimage.metrics import peak_signal_noise_ratio
import matplotlib.pyplot as plt

noisy_img = img_as_float(io.imread("./Img/LDCT_Noisy.jpeg"))
ref_img = img_as_float(io.imread("./Img/LDCT_Clean.jpeg"))
wavelet_smoothed = denoise_wavelet(noisy_img, channel_axis=-1, method='BayesShrink', mode='soft', rescale_sigma=True)
noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
Wavelet_cleaned_psnr = peak_signal_noise_ratio(ref_img, wavelet_smoothed)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", Wavelet_cleaned_psnr)
plt.imsave("./Img/wavelet_smoothed.jpeg", wavelet_smoothed, cmap='gray')