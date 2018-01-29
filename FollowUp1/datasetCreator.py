import cv2 as cv

cascade_face = cv.CascadeClassifier('C:\\Users\\Dell\\Desktop\\Python\\FollowUp\\FollowUp1\\haarcascade_frontalface_default.xml')
camera = cv.VideoCapture(0)  # video capture

id = input('Enter user id: ')
samp = 0

while True:
    ret, img = camera.read()     # ret ??
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)   # convert to gray img
    face = cascade_face.detectMultiScale(gray, 1.3, 5)  # (img, scaleFactor, minNeighbors)
    for (x, y, w, h) in face:
        samp = samp+1
        cv.imwrite('C:\\Users\\Dell\\Desktop\\Python\\FollowUp\\FollowUp1\\dataSet\\user.'+str(id)+'.'+str(samp)+'.jpg', gray[y:y+h, x:x+w])
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv.waitKey(100)
    cv.imshow("Face", img)
    cv.waitKey(1)
    # if cv.waitKey(1) == ord('q'):
    #     break
    if samp > 20:
        break
camera.release()
cv.destroyAllWindows()

