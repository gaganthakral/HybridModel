import pydicom
import numpy
import matplotlib.pyplot as plt
file_path = './Img/1-001.dcm'
medical_image = pydicom.read_file(file_path)
print(medical_image)
image=medical_image.pixel_array
print(image.shape)
plt.imshow(image, cmap='gray')
plt.show()
print(image.min())
print(image.max())
print(dir(medical_image))