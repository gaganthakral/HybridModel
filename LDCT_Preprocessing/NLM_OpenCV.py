import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_ubyte, img_as_float
from matplotlib import pyplot as plt
from skimage import io
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

noisy_img = img_as_float(io.imread("./Img/LDCT_Noisy.jpeg",as_gray=True))
ref_img = img_as_float(io.imread("./Img/LDCT_Clean.jpeg"))

# fastNlMeansDenoising(InputArray src, OutputArray dst, float h=3, int templateWindowSize=7, int searchWindowSize=21 )

NLM_CV2_denoise_img = cv2.fastNlMeansDenoising(noisy_img, None, 15, 7, 21)


plt.imsave("./Img/NLM_CV2_denoised.jpeg", NLM_CV2_denoise_img, cmap='gray')
