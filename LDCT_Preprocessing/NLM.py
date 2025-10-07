from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_ubyte, img_as_float
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
from skimage.metrics import peak_signal_noise_ratio


noisy_img = img_as_float(io.imread("./Img/LDCT_Noisy.jpeg"))
ref_img = img_as_float(io.imread("./Img/LDCT_Clean.jpeg"))

sigma_est = np.mean(estimate_sigma(noisy_img, channel_axis=-1))


NLM_skimg_denoise_img = denoise_nl_means(noisy_img, h=1.15 * sigma_est, fast_mode=True, patch_size=9, patch_distance=5)

noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
NLM_skimg_cleaned_psnr = peak_signal_noise_ratio(ref_img, NLM_skimg_denoise_img)
print("PSNR of input noisy image = ", noise_psnr)
print("PSNR of cleaned image = ", NLM_skimg_cleaned_psnr)


denoise_img_as_8byte = img_as_ubyte(NLM_skimg_denoise_img)

#plt.imshow(NLM_skimg_denoise_img)
#plt.imshow(denoise_img_as_8byte, cmap=plt.cm.gray, interpolation='nearest')
plt.imsave("./Img/NLM_skimage_denoised.jpeg", denoise_img_as_8byte, cmap='gray')