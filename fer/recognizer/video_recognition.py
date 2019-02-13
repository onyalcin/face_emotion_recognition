import os

import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array

import numpy as np

from .recognizer import VideoRecognizer


class EmoVideoRecognizer(VideoRecognizer):
    def __init__(self):
        # parameters for loading data and images
        detection_model_path = os.path.join('fer', 'trained_models',
                                            'detection_models', 'haarcascade_frontalface_default.xml')
        emotion_model_path = os.path.join('fer', 'trained_models',
                                          'emotion_models', 'fer2013_mini_XCEPTION.102-0.66.hdf5')
        # loading models
        self.face_detection = cv2.CascadeClassifier(detection_model_path)
        self.emotion_classifier = load_model(emotion_model_path, compile=False)
        self.emotion_classifier._make_predict_function()

        # no contempt bu extra neutral
        self.EMOTIONS = ["anger", "disgust", "fear", "joy", "sad", "surprise", "neutral"]

    def recognize(self, frame):
        frame = cv2.resize(frame, (300, 200))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detection.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        if len(faces) == 0:
            return {}

        faces = sorted(faces, reverse=True,
                       key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        # Extract ROI from the grayscale image,
        # resize, and then prepare it for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        predictions = self.emotion_classifier.predict(roi)[0]
        #emotion_probability = np.max(predictions)
        #label = self.EMOTIONS[predictions.argmax()]

        # dict containing emotions and probabilities in 0.0f
        emotions = dict(zip(self.EMOTIONS, predictions))
        # returning emotions as a separate dict
        return {"emotions": emotions}
