"""Mirror printed output to a log file and always restore stdout."""

from contextlib import contextmanager
import sys


class Tee:
    """File-like object that writes to several streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
        return len(message)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_stdout(path):
    """Print to the terminal and to path. The file is closed and stdout restored on exit."""
    original = sys.stdout
    with open(path, "w", encoding="utf-8") as log:
        sys.stdout = Tee(original, log)
        try:
            yield
        finally:
            sys.stdout = original
