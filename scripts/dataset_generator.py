"""The script that generates all 3 datasets and their features"""
import numpy as np
import pandas as pd
import features as features
from typing import List, TypeVar, Tuple, Union
import sys, getopt, math, pytz, pickle

T = TypeVar('T')

MAX_ROWS = None # None = infinite

TRAINING_SET_PERCENTAGE = 70
REPORT_SIZE = 100
BATCH_SIZE = 32
MIN_GROUP_SIZE = 150
MIN_GROUP_SIZE = max(MIN_GROUP_SIZE, (BATCH_SIZE * 2) + 2)

class Row:
	"""A row of data"""

	def __init__(self, row: list):
		row_one_split = row[1].split("@")
		row_two_split = row[2].split("@")

		self._row = row
		self.time = pytz.utc.localize(row[0].to_pydatetime()).timestamp()
		self.source_user = self.user = row_one_split[0]
		self.domain = row_one_split[1]
		self.dest_user = row_two_split[0]
		self.src_computer = row[3]
		self.dest_computer = row[4]
		self.auth_type = row[5]
		self.logon_type = row[6]
		self.auth_orientation = row[7]
		self.status = row[8]

	def to_str(self) -> str:
		"""Converts the row to a string"""
		return str(self._row)


class PropertyDescription:
	def __init__(self):
		self._list = list()
		self._counts = dict()
		self._unique = 0
		self._freq = 0


	def append(self, item: str):
		"""Appends given item to the list of the property"""
		if item not in self._list:
			self._list.append(item)
			self._unique += 1

		self._counts[item] = self._counts.get(item, 0) + 1

	@property
	def unique(self) -> int:
		return len(self._list)

	@property
	def freq(self) -> int:
		highest_index = 0
		for key, value in self._counts.items():
			if value > highest_index:
				highest_index = value

		return highest_index

	@property
	def list(self) -> List[str]:
		return self._list


class Features:
	"""All the features fr a model"""

	def __init__(self):
		self._current_access = 0
		self._last_access = 0
		self._domains = PropertyDescription()
		self._dest_users = PropertyDescription()
		self._src_computers = PropertyDescription()
		self._dest_computers = PropertyDescription()
		self._failed_logins = 0
		self._login_attempts = 0


	def update_dest_users(self, user: str):
		"""Updates the dest_users list"""
		if user != "?":
			self._dest_users.append(user)

	def update_src_computers(self, computer: str):
		"""Updates the src_computers list"""
		if computer != "?":
			self._src_computers.append(computer)

	def update_dest_computers(self, computer: str):
		"""Updates the dest_computers list"""
		if computer != "?":
			self._dest_computers.append(computer)

	def update_domains(self, domain: str):
		"""Updates the dest_users list"""
		if domain != "?":
			self._domains.append(domain)

	def update(self, row: Row):
		"""Updates all data lists for this feature class"""
		self.update_dest_users(row.dest_user)
		self.update_src_computers(row.src_computer)
		self.update_dest_computers(row.dest_computer)
		self.update_domains(row.domain)

		self._last_access = self._current_access
		self._current_access = row.time
		if row.status != 'Success':
			self._failed_logins += 1
		self._login_attempts += 1

	@property
	def last_access(self) -> int:
		"""The last time this user has authenticated themselves"""
		return self._last_access

	@property
	def current_access(self) -> int:
		"""The timestamp of the current auth operation"""
		return self._current_access

	@property
	def dest_users(self) -> PropertyDescription:
		"""All destination users"""
		return self._dest_users

	@property
	def src_computers(self) -> PropertyDescription:
		"""All source computers"""
		return self._src_computers

	@property
	def dest_computers(self) -> PropertyDescription:
		"""All destination computers"""
		return self._dest_computers

	@property
	def domains(self) -> PropertyDescription:
		"""All domains accessed"""
		return self._domains

	@property
	def percentage_failed_logins(self) -> float:
		"""The percentage of non-successful logins"""
		return self._failed_logins / self._login_attempts

	def get_time_since_last_access(self) -> int:
		"""Gets the time between the current access and the last one"""
		return self._current_access - self._last_access


def normalize_all(features: List[float]):
	np_arr = np.array(features)
	reshaped = np.reshape(features, (np_arr.shape[1], np_arr.shape[0]))

	maxes = list()
	for col in range(len(reshaped)):
		col_max = max(reshaped[col])
		reshaped[col] = [float(i)/col_max for i in reshaped[col]]
		maxes.append(col_max)

	return np.reshape(np.append(maxes,
			np.reshape(np.array(reshaped), (np_arr.shape[0], np_arr.shape[1]))),
			(np_arr.shape[0] + 1, np_arr.shape[1]))



def convert_to_features(group):
	"""This converts given group to a list of features
	:type group: pd.Group
	"""
	feature_list = list()
	current_features = Features()

	for row in group.itertuples():
		row = Row(row)
		current_features.update(row)
		feature_list.append(features.extract(row, current_features))

	return normalize_all(feature_list)


def closest_multiple(target: int, base: int) -> int:
	lower_bound = target - (target % base)
	if float(target - lower_bound) > (base / 2):
		# Round up
		return lower_bound + base
	return lower_bound


def split_list(target: List[T], batch_size: int=1) -> Union[Tuple[List[T], List[T]], bool]:
	"""This splits given list into a distribution set by the *_SET_PERCENTAGE consts"""
	target_length = len(target)

	# Attempt to account for batch sizes already
	training_set_length = closest_multiple(int(math.ceil(
		(TRAINING_SET_PERCENTAGE / 100) * float(target_length)
	)), batch_size) + 1

	test_set_length = (target_length - 1) - training_set_length
	test_set_length = test_set_length - (test_set_length % batch_size)

	if test_set_length == 0:
		training_set_length -= batch_size
		test_set_length += batch_size

	test_set_length += 1

	if training_set_length <= 1 or test_set_length <= 1:
		return False

	return target[0:training_set_length], \
		   target[training_set_length:training_set_length + test_set_length]


def split_dataset(group) -> Tuple[List[List[float]], Union[Tuple[List[float], List[float]], bool]]:
	"""This converts the dataset to features and splits it into 3 parts
	:type group: pd.Group
	"""
	feature_data = convert_to_features(group)
	return feature_data[0:1], split_list(feature_data[1:], BATCH_SIZE)


def get_io_settings(argv: sys.argv) -> Tuple[str, str]:
	"""This gets the input and output files from the command line arguments"""
	input_file = '/data/s1481096/LosAlamos/data/auth_small.h5'
	output_file = '/data/s1495674/features.p'

	try:
		opts, args = getopt.getopt(argv, 'i:o:')
	except getopt.GetoptError:
		print("Command line arguments invalid")
		sys.exit(2)

	for opt, arg in opts:
		if opt == '-i':
			input_file = arg
		elif opt == '-o':
			output_file = arg
		else:
			print("Unrecognized argument passed")
			sys.exit(2)

	return input_file, output_file


def main(argv: sys.argv):
	"""The main function"""
	users_list = list()

	input_file, output_file = get_io_settings(argv)

	print("Gathering features for", MAX_ROWS if MAX_ROWS is not None else "as many as there are" , "rows",
		  "using a batch size of", BATCH_SIZE)

	rows = 0
	f = pd.read_hdf(input_file, 'auth_small', start=0, stop=MAX_ROWS) \
			.groupby(["source_user"])
	file_length = len(f)
	for name, group in f:
		if len(group.index) > MIN_GROUP_SIZE:
			user_name = group.iloc[0].get('source_user').split('@')[0]

			if user_name == "ANONYMOUS LOGON" or user_name == "ANONYMOUS_LOGON":
				continue

			maxes_row, split_dataset_result = split_dataset(group)
			if split_dataset_result:
				training_set, test_set = split_dataset_result
				user = {
					"user_name": user_name,
					"maxes": maxes_row[0],
					"datasets": {
						"training": training_set,
						"test": test_set
					}
				}
				users_list.append(user)

			rows += 1

			if rows % REPORT_SIZE == 0:
				sys.stdout.write('At row ')
				sys.stdout.write(str(rows))
				sys.stdout.write('/~')
				sys.stdout.write(str(file_length))
				sys.stdout.write('\n')

	print("Did a total of", len(users_list), "users")
	print("Done gathering data, outputting to file", output_file)
	output = open(output_file, 'wb')
	pickle.dump(users_list, output)
	sys.exit()


if __name__ == "__main__":
	main(sys.argv[1:])
