
import cv2
from scipy import misc
import imageio  

img_path = "200467.jpg"
img1 = misc.imread(img_path)
print(img1.shape)
img2 = cv2.imread(img_path, -1) 
print(img2.shape)

img3 = imageio.imread(img_path)
print(img3.shape)