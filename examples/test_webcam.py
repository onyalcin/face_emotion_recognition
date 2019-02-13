from concurrent.futures import ThreadPoolExecutor
from fer.video_input.video_thread import VideoInput
from fer.recognizer.video_recognition import EmoVideoRecognizer

import cv2
import numpy as np


def recognize_from_webcam():
    emo_rec = EmoVideoRecognizer()

    def on_frame(frame):
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        clone = frame.copy()

        predictions = emo_rec.recognize(frame)["emotions"]
        recognition = max(predictions, key=predictions.get)

        for (i, (emotion, prob)) in enumerate(predictions.items()):
            # construct the label text
            text = "{}: {:.2f}%".format(emotion, prob * 100)

            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5),
                          (w, (i * 35) + 35), (0, 0, 255), -1)
            if emotion == recognition:
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 0, 0), 2)
            else:
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 2)
        cv2.imshow('Web-cam', clone)
        cv2.imshow("Results", canvas)
        cv2.waitKey(1)

    with ThreadPoolExecutor() as executor:
        with VideoInput(executor=executor) as vi:
            vi.start(on_frame)
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    recognize_from_webcam()
