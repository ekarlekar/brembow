import time


class Timer:

    def __init__(self, name):
        self.name = name
        self._elapsed = 0
        self._start = 0

    def start(self):
        self._start = time.time()

    def stop(self):
        self._elapsed += (time.time() - self._start)

    def __repr__(self):
        return "%s\t: %.3fs" % (self.name, self._elapsed)
