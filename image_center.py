import cv2 
import numpy as np
import os
import math
image = cv2.imread('/home/krshna/Downloads/test.jpeg')
height, width, channels = image.shape 

h=height/2
w=width/2

new_img=image[h:h+h,w:w+w]
path = '/home/krshna/Desktop/object-sementation/seg_imgs'
cv2.imwrite('/home/krshna/Desktop/object-sementation/seg_imgs/s1'+'.jpg', new_img)

new_img=image[0:0+h,0:0+w]
path = '/home/krshna/Desktop/object-sementation/seg_imgs'
cv2.imwrite('/home/krshna/Desktop/object-sementation/seg_imgs/s2'+'.jpg', new_img)

new_img=image[0:0+h,w:w+w]
path = '/home/krshna/Desktop/object-sementation/seg_imgs'
cv2.imwrite('/home/krshna/Desktop/object-sementation/seg_imgs/s3'+'.jpg', new_img)

new_img=image[h:h+h,0:0+w]
path = '/home/krshna/Desktop/object-sementation/seg_imgs'
cv2.imwrite('/home/krshna/Desktop/object-sementation/seg_imgs/s4'+'.jpg', new_img)

'''print height
print width'''
