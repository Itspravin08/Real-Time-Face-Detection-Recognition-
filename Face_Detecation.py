# import required packages

import cv2
import numpy as np


# Get Pre-trained classifiers for Detecte face
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Capture frame-by-frame
cap = cv2.VideoCapture(0)
cap.set(3,640)  # Set video Width
cap.set(4,480)  # Set Video Heigth

while True:
    # Find haar cascade to draw bounding box around face
    ret , img = cap.read()

    gary = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)  # Convert to gary scale

    # detect faces available on camera
    faces = faceCascade.detectMultiScale(gary ,
                                         scaleFactor=1.2,
                                         minNeighbors=5,
                                         minSize=(20,20)
                                         )

    # take each face available on the camera and Preprocess it
    # Draw a rectangle around the face
    for (x ,y ,w ,h) in faces:
        cv2.rectangle(img,(x,y) , ( x+w , y+h ) , (255,0,0) , 2)
        roi_gray = gary[y:y+h , x:x+w]
        roi_Color = img[y:y+h ,x:x+w]

    # Display the resulting frame
    cv2.imshow("video",img)

    k = cv2.waitKey(30) # & Oxff
    if k == 27: # Press 'Esc' to quit
        break

cap.release()
cv2.destroyALLWindows()