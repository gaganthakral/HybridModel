import matplotlib.pyplot as plt

from skimage.restoration import denoise_wavelet, cycle_spin
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
from skimage import io


noisy_img = img_as_float(io.imread("./Img/LDCT_Noisy.jpeg"))
ref_img = img_as_float(io.imread("./Img/LDCT_Clean.jpeg"))


denoise_kwargs = dict(channel_axis=-1, wavelet='db1', method='BayesShrink', rescale_sigma=True)

all_psnr = []
max_shifts = 3

Shft_inv_wavelet = cycle_spin(noisy_img, func=denoise_wavelet, max_shifts = max_shifts, func_kw=denoise_kwargs, channel_axis=-1)

noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
shft_cleaned_psnr = peak_signal_noise_ratio(ref_img, Shft_inv_wavelet)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", shft_cleaned_psnr)

plt.imsave("./Img/Shift_Inv_wavelet_smoothed.jpeg", Shft_inv_wavelet, cmap='gray')

