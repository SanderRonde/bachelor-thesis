"""Estimates how long operations may take"""
import time
import math


class Timer:
	"""A timer to determine how long the entire operation might take"""

	def __init__(self, maximum: int):
		self._max = maximum
		self._current = 0
		self.start_time = time.time()

	def add_to_current(self, num: int):
		"""Adds another **num** to the current progress"""
		self._current += num

	@staticmethod
    def format_time(seconds: int) -> Tuple[Union[int, None], Union[int, None], Union[int, None]]:
		"""Turns seconds into hours, mins and seconds"""
        if seconds <= 60:
            return None, None, seconds

        mins = math.floor(seconds / 60)
        seconds = seconds % 60

        if mins <= 60:
            return None, round(mins), round(seconds)

        hours = math.floor(mins / 60)
        mins = mins % 60
        return round(hours), round(mins), round(seconds)

    @staticmethod
    def stringify_time(passed_time: Tuple[Union[int, None], Union[int, None], Union[int, None]]) -> str:
		"""Turns hours, mins and seconds into a string format"""
        hours, mins, seconds = passed_time

        if mins is not None:
            if hours is not None:
                return str(hours) + 'h' + str(mins) + 'm' + str(seconds) + 's'
            else:
                return str(mins) + 'm' + str(seconds) + 's'
        else:
            return str(seconds) + 's'

    def get_eta(self) -> str:
		"""Gets the ETA given current progress"""
        if self._current == 0:
            return 'unknown'

        passed_time = time.time() - self.start_time
        amount_done = self._current / self._max

        time_remaining = round(((1 / amount_done) * passed_time) - passed_time)
        return self.stringify_time(self.format_time(time_remaining))

    def report_total_time(self) -> str:
		"""Reports the total time since timer initialization"""
        return self.stringify_time(self.format_time(time.time() - self.start_time))


	@property
	def current(self):
		"""Gets the current passed actions"""
		return self._current
