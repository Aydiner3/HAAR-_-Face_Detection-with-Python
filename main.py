import cv2
import matplotlib.pyplot as plt

img = cv2.imread("faceDetection/yuz.jpg" , 0)
plt.figure()
plt.imshow(img, cmap="gray")
plt.show()

face_cascade = cv2.CascadeClassifier("faceDetection/haarcascade_frontalface_default.xml")
face_rect = face_cascade.detectMultiScale(img, minNeighbors = 4)

for (x,y,w,h) in face_rect:
    cv2.rectangle(img , (x,y), (x+w , y+h), (255,255,255), 10)

plt.figure()
plt.imshow(img , cmap="gray") 
plt.axis("off")  
plt.show()


