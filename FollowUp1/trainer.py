import cv2 as cv,os
import numpy as np
from PIL import Image

recognizer = cv.face.LBPHFaceRecognizer_create()
path = 'C:\\Users\\Dell\\Desktop\\Python\\FollowUp\\FollowUp1\\dataSet'


def get_image_with_id(path):

    ImagesPaths = [os.path.join(path, f) for f in os.listdir(path)]

    faces = []
    IDs = []

    for imagepath in ImagesPaths:
        # read image and convert it into gray scale
        cvtgray = Image.open(imagepath).convert('L')
        # convert image format into numpy array
        # 'uint8' is a dataType
        cvtNp = np.array(cvtgray, 'uint8')
        # get the id's of image
        ID = int(os.path.split(imagepath)[-1].split('.')[1])

        faces.append(cvtNp)
        IDs.append(ID)

        cv.imshow('Training', cvtNp)
        cv.waitKey(10)

    print(IDs)
    print(faces)
    return np.array(IDs), faces


IDs, faces = get_image_with_id(path)
recognizer.train(faces, IDs)
recognizer.write('C:\\Users\\Dell\\Desktop\\Python\\FollowUp\\FollowUp1\\recognizer\\trainingData.yml')
cv.destroyAllWindows()



