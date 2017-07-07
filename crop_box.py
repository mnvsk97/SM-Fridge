import cv2
with open('/home/krshna/Desktop/boxes/darknet/box_cord.txt', 'r') as file:        
	rows = [[int(x) for x in line.split(' ')[:-1]] for line in file] 
img = cv2.imread("/home/krshna/Downloads/test.jpeg")  
for i in range(len(rows)):
	y1=(rows[i][0])
	y2=(rows[i][1])
	x1=(rows[i][2])
	x2=(rows[i][3])
	roi = img[y1:y2,x1:x2]
	#cv2.imshow("sulli",roi)
	cv2.imwrite("/home/krshna/Desktop/segmented_images/tests"+str(i)+".jpeg", roi)