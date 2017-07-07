import cv2 
import numpy as np
import os
import math

image = cv2.imread("/home/krshna/Downloads/test.jpeg")

#cv2.waitKey(0)

#gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

imgThresh = cv2.medianBlur(image, 9)             #Median blur for redcuing the salt and pepper noise
#ret3, imgThresh = cv2.threshold(imgThresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # blur


edged = cv2.Canny(imgThresh, 10, 250)

#(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

edged, cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



def smaller(a, b):       # function definition for sorting
              areaA = cv2.contourArea(a)    # based on the area of contour
              areaB = cv2.contourArea(b)
              if areaA > areaB:
                return -1
              if areaA == areaB:
                return 0
              else:
                return 1

cnts.sort(smaller)  # sorting contours based on its area

idx = 0
for c in cnts:

	if (cv2.contourArea(c)>110):
          
          x,y,w,h = cv2.boundingRect(c)
      	  if w>10 and h>10:
      			idx+=1
      			new_img=image[y:y+h,x:x+w]
      			path = '/home/krshna/Desktop/object-sementation/seg_imgs'
      			cv2.imwrite('/home/krshna/Desktop/object-sementation/seg_imgs/s'+str(idx)+'.jpg', new_img)



cv2.imshow("im",image)

for c in cnts:
    if (cv2.contourArea(c)>100):
    	rect0 = cv2.minAreaRect(c)    #setting minimum bounding area rectangle on the contour
        box = cv2.boxPoints(rect0)     #storing the cente,dimensions and R_angle in rect_0
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0,0,255),2)
        


    

cv2.imshow("output",image)
cv2.waitKey(0)

