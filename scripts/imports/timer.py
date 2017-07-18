import time
import math


class Timer:
    """A timer to determine how long the entire operation might take"""

    def __init__(self, maximum: int):
        self._max = maximum
        self._current = 0
        self.start_time = time.time()

    def add_to_current(self, num: int):
        self._current += num

    def get_eta(self) -> str:
        if self._current == 0:
            return 'unknown'

        passed_time = time.time() - self.start_time
        amount_done = self._current / self._max

        seconds = round(((1 / amount_done) * passed_time) - passed_time)
        if seconds <= 60:
            return str(seconds) + 's'

        mins = math.floor(seconds / 60)
        seconds = seconds % 60

        if mins <= 60:
            return str(mins) + 'm' + str(seconds) + 's'

        hours = math.floor(mins / 60)
        mins = mins % 60
        return str(hours) + 'h' + str(mins) + 'm' + str(seconds) + 's'

    @property
    def current(self):
        return self._current
