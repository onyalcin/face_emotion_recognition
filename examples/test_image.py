from fer.recognizer.video_recognition import EmoVideoRecognizer

import os
import glob
import cv2


def recognize_image():
    emo_frame_path = os.path.join('data', 'images')
    frame_names = glob.glob(emo_frame_path)

    emo_rec = EmoVideoRecognizer()

    for frame_path in frame_names:
        frame = cv2.imread(frame_path)

        prediction = emo_rec.recognize(frame)
        print(frame_path + ': ' + str(prediction))


if __name__ == '__main__':
    recognize_image()
