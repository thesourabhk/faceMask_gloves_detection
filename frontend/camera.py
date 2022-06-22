import cv2

from imutils.video import WebcamVideoStream
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as k
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import cv2
import datetime


class Video(object):
    def __init__(self):
        self.stream = WebcamVideoStream(src=0).start()
        self.detector = cv2.CascadeClassifier(
            '../haarcascade_frontalface_default.xml')
        self.mymodel = load_model('../mymodel.h5')

    def __del__(self):
        self.stream.stop()

    def get_frame(self):
        img = self.stream.read()

        face = self.detector.detectMultiScale(img, 1.1, 4)
        for (x, y, h, w) in face:
            face_img = img[y:y+h, x:x+w]
            cv2.imwrite('../temp.jpg', face_img)
            test_image = image.load_img(
                '../temp.jpg', target_size=(150, 150, 3))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            pred = self.mymodel.predict(test_image)[0][0]
            # print(pred)
            if pred == 1:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(img, 'NO MASK', ((x+w)//2, y+h+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(img, 'MASK', ((x+w)//2, y+h+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            datet = str(datetime.datetime.now())
            cv2.putText(img, datet, (400, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        ret, jpg = cv2.imencode('.jpg', img)
        data = []
        data.append(jpg.tobytes())
        return data
