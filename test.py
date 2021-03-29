from PIL import Image
import numpy as np
import cv2

img = cv2.imread("/Users/albi/Desktop/spineeee.png", flags=0)
img_gray= Image.fromarray(img, mode='L')
imgRGB = img_gray.convert('RGB')
imgRGB.putalpha(img_gray)
image = imgRGB.convert('RGBA')

image.save("/Users/albi/Desktop/spinalpha.png")


