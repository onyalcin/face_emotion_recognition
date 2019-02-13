# Emotion Recognition from Facial Expressions

This package is a submodule of the "Empathy in Embodied
Conversational Agents" project in iVizLab @Simon Fraser University.

It is intended to provide a simple emotion recognition component
to capture the facial emotions from offline images, videos as well as online web-cam data.


### Prerequisites
The system is tested for:

- Python == 3.6.*

- opencv-python >= 3.4, <=4.0

- keras >= 2.0, <=2.2.* with tensorflow = 1.12.0  backend

### Installing
The package and dependencies can be installed via pip with:
```
pip install git+https://github.com/o-n-yalcin/face_emotion_recognition
```

Or clone the project and:

```
$ git clone https://github.com/o-n-yalcin/face_emotion_recognition.git
$ pip install -e ./face_emotion_recognition
```

You can start using the recognizer. You only need to pass a frame for the recognizer to do its job:
```
from fer.recognizer.video_recognition import EmoVideoRecognizer

emo = EmoVideoRecognizer()
predictions = emo.recognize(frame)
```

## Example Usages
The project includes three examples within the examples folder.
Offline examples require video and image files to be placed in
the related folders within the data folder.

**test_image.py** prints out the prediction results for each image in the data/images folder.
```
$ python -m examples.test_image
```

**test_video_file** creates a csv file of predictions per frame
for every video within the data/videos folder.
```
$ python -m examples.test_video_file
```

Online example creates a thread for webcam input and shows the prediction results in real time.
```
$ python -m examples.test_webcam
```

Following these examples you can create your own version according to your needs.

## Authors

* **Ozge Nilay Yalcin** - [ony](https://github.com/o-n-yalcin)

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details
