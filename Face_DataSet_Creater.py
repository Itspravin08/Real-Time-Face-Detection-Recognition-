# import required packages

import cv2
import numpy as np


# Get Pre-trained classifiers for Detecte face
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Capture frame-by-frame
cap = cv2.VideoCapture(0)
cap.set(3,640)  # Set video Width
cap.set(4,480)  # Set video Heigth

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")

# Initialize individual sampling face count
count = 0

while True:
    # Capture frame-by-frame
    ret , img = cap.read()

    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

    # detect faces available on camera Using haarcascade Classifier
    faces = face_detector.detectMultiScale(gray ,
                                         scaleFactor=1.2,
                                         minNeighbors=5,
                                         )

    # Draw a rectangle around the faces
    for (x ,y ,w ,h) in faces:

        cv2.rectangle(img,(x,y) , ( x+w , y+h ) , (255,0,0) , 2)

        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("E:/ML/Real-time-face-recognition/Datasets/User." + str(face_id) + '.' +str(count) + ".jpg", gray[y:y + h, x:x + w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30:  # Take 30 face sample and stop video
        break



# When everything is done, release the capture
print("\n [INFO] Exiting Program and cleanup stuff")
cap.release()
cv2.destroyAllWindows()
