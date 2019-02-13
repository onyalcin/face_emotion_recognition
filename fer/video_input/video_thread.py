import logging
import cv2
import threading


logger = logging.getLogger(__name__)


class VideoInput:
    def __init__(self, executor):
        self._executor = executor
        self._video_capture = None
        self._stop_requested = threading.Event()
        self._future = None

    def __enter__(self):
        self._video_capture = cv2.VideoCapture(0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._future:
            self.stop().result(timeout=1)
        self._video_capture.release()
        self._video_capture = None

    def start(self, callback):
        assert not self._future, 'Thread is already running'
        self._future = self._executor.submit(self._run, callback)

    def stop(self):
        assert self._future, 'Thread is not running'
        self._stop_requested.set()
        f = self._future
        self._future = None
        return f

    def _run(self, callback):
        while not self._stop_requested.is_set():
            ret, frame = self._video_capture.read()
            if ret:
                callback(frame)
