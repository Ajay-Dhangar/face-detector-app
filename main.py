import cv2 as cv

trained_face_data = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv.imread("./images/Ajay.jpg")

gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

face_coordinates = trained_face_data.detectMultiScale(gray_img)

for (x, y, w, h) in face_coordinates:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv.imshow("Face Detector", img)
cv.waitKey()

print("Code Completed")