import matplotlib.pyplot as plt
import pydicom

file_path = './Img/1-020.dcm'
medical_image = pydicom.dcmread(file_path)
image = medical_image.pixel_array
plt.imshow(image, cmap='gray')
plt.imsave('./Img/LDCT_Clean.jpeg', image, cmap='gray')
