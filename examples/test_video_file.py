from fer.recognizer.video_recognition import EmoVideoRecognizer

import glob
import os
import cv2

import csv


def recognize_from_video_file():
    emo_videos_path = os.path.join('data', 'videos')
    headers = ['time', 'anger', 'disgust', 'fear', 'joy', 'sad', 'surprise', 'neutral']
    emo_rec = EmoVideoRecognizer()

    video_names = glob.glob(emo_videos_path+'\*.mp4')
    for video in video_names:
        print(video)
        cap = cv2.VideoCapture(video)
        with open(video[:-4]+'.csv', 'a') as output_file:
            outf = csv.DictWriter(output_file, fieldnames=headers)
            outf.writeheader()
            count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret == 1:
                    try:
                        emotions = emo_rec.recognize(frame)['emotions']
                    except:
                        emotions = dict.fromkeys(headers[1:], None)
                    outf.writerow({**emotions, **{'time': count}})
                    count += 1
                else:
                    break
        cap.release()


if __name__ == '__main__':
    recognize_from_video_file()