import numpy as np
import cv2
from matplotlib import pyplot as plt
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
plt.figure()
cap1 = cv2.VideoCapture("godfather.mp4")
cap2 = cv2.VideoCapture(0)

while 1:
    ret, img = cap1.read()
    ret2, img2 = cap2.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    faces1 = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)
    #face1_value=set(faces1)


    for (x,y,w,h) in faces1:

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)# for first video
        # for first video
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
                
        eyes = eye_cascade.detectMultiScale(roi_gray)# for first 
        
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            #cv2.rectangle(roi_color2,(ex2,ey2),(ex2+ew2,ey2+eh2),(0,255,0),2)

    for (x2,y2,w2,h2) in faces2:
        cv2.rectangle(img2,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)# for second video
        # for second video
        roi_gray2 = gray2[y2:y2+h2, x2:x2+w2]
        roi_color2 = img2[y2:y2+h2, x2:x2+w2]
        eyes2 = eye_cascade.detectMultiScale(roi_gray2)# for second

        for (ex2,ey2,ew2,eh2) in eyes2:

            cv2.rectangle(roi_color2,(ex2,ey2),(ex2+ew2,ey2+eh2),(0,255,0),2)
    cv2.imshow('img',img)
    cv2.imshow('img2',img2)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()