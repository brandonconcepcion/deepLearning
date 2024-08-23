import cv2 as cv 
import numpy as np 

blank = np.zeros((500,500, 3), dtype = 'uint8')
cv.imshow('Blank Image', blank)

#1. Paint the image a certain color 
blank[200:300, 300:400] = 0,255,0
cv.imshow('Green', blank)

#2 Draw A rectangle 
cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2) , (0, 255, 0), thickness = -1 )
cv.imshow('Rectangle', blank)

#3 Drawing a Circle
cv.circle(img=blank, center=(250, 250), radius=40, color=(255,0,0),thickness=-1)
cv.imshow('Circle', blank)

#4 Draw a line
cv.line(img=blank, pt1=(0,0), pt2=(250,250), color=(255,255,255), thickness=2)
cv.imshow('Line', blank)

#5 Write Text on image 
cv.putText(img=blank, text="Hi! I'm Brandon", org= (50,225), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1.0,
           color=(0,255,0), thickness=2)
cv.imshow('Text', blank)


cv.waitKey(0)

