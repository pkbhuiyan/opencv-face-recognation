import cv2

cascade_face = cv2.CascadeClassifier('C:\\Users\\Dell\\Desktop\\Python\\FollowUp\\FollowUp1\\haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)
# to recognize face from Training dataSet
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:\\Users\\Dell\\Desktop\\Python\\FollowUp\\FollowUp1\\recognizer\\trainingData.yml')

id = 0
# font = cv2.cv2.InitFont(cv2.cv2.CV_FONT_HERSHEY_COMPLEX_SMALL, 5, 1, 0, 4)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

while True:

    check, img = camera.read()

    # convert BGR to GRAY
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade_face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        # check confidence with recognizer
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        confidence = round(confidence, 2)
        cv2.putText(img, str(id), (x, y+h), font, 1, (0, 0, 255), 2)
        cv2.putText(img, "SURE:"+str(confidence)+"%", (x, y+h+20), font, 1, (0, 0, 255), 2)


    cv2.imshow("Detect", img)
    # close win when 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
