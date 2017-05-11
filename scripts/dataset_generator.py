"""The script that generates all 3 datasets and their features"""
import pandas as pd
import features as features
import sys, getopt, math, pytz, pickle

MAX_ROWS = None # None = infinite
MIN_GROUP_SIZE = 10

TRAINING_SET_PERCENTAGE = 70
TEST_SET_PERCENTAGE = 30


class Row:
	"""A row of data"""

	def __init__(self, row):
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

	def to_str(self):
		# type: () -> str
		"""Converts the row to a string"""
		return self._row


class Features:
	"""All the features fr a model"""

	def __init__(self):
		self._current_access = 0
		self._last_access = 0
		self._domains = list()
		self._dest_users = list()
		self._src_computers = list()
		self._dest_computers = list()

	def update_dest_users(self, user):
		"""Updates the dest_users list"""
		if user != "?" and user not in self._dest_users:
			self._dest_users.append(user)

	def update_src_computers(self, computer):
		"""Updates the src_computers list"""
		if computer != "?" and computer not in self._src_computers:
			self._src_computers.append(computer)

	def update_dest_computers(self, computer):
		"""Updates the dest_computers list"""
		if computer != "?" and computer not in self._dest_computers:
			self._dest_computers.append(computer)

	def update_domains(self, domain):
		"""Updates the dest_users list"""
		if domain != "?" and domain not in self._domains:
			self._domains.append(domain)

	def update(self, row):
		"""Updates all data lists for this feature class"""
		self.update_dest_users(row.dest_user)
		self.update_src_computers(row.src_computer)
		self.update_dest_computers(row.dest_computer)
		self.update_domains(row.domain)

		self._last_access = self._current_access
		self._current_access = row.time

	@property
	def last_access(self):
		# type: () -> int
		"""The last time this user has authenticated themselves"""
		return self._last_access

	@property
	def current_access(self):
		# type: () -> int
		"""The timestamp of the current auth operation"""
		return self._current_access

	@property
	def dest_users(self):
		# type: () -> list
		"""All destination users"""
		return self._dest_users

	@property
	def src_computers(self):
		# type: () -> list
		"""All source computers"""
		return self._src_computers

	@property
	def dest_computers(self):
		# type: () -> list
		"""All destination computers"""
		return self._dest_computers

	@property
	def domains(self):
		# type: () -> list
		"""All domains accessed"""
		return self._domains

	def get_time_since_last_access(self):
		# type: () -> int
		"""Gets the time between the current access and the last one"""

		return self._current_access - self._last_access


def convert_to_features(group):
	"""This converts given group to a list of features"""
	# type: (pd.Group) -> list
	feature_list = list()
	current_features = Features()

	for row in group.itertuples():
		row = Row(row)
		current_features.update(row)
		feature_list.append(features.extract(row, current_features))

	return feature_list


def split_list(target):
	"""This splits given list into a distribution set by the *_SET_PERCENTAGE consts"""
	target_length = len(target)

	training_set_end = math.ceil((TRAINING_SET_PERCENTAGE / 100) * target_length)

	return target[0:training_set_end], \
		   target[training_set_end:]


def split_dataset(group):
	"""This converts the dataset to features and splits it into 3 parts"""
	feature_data = convert_to_features(group)
	return split_list(feature_data)


def get_io_settings(argv):
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


def main(argv):
	"""The main function"""
	users_list = list()

	input_file, output_file = get_io_settings(argv)

	print("Gathering features for", MAX_ROWS if MAX_ROWS is not None else "as many as there are" , "rows")

	for name, group in pd.read_hdf(input_file, 'auth_small', start=0, stop=MAX_ROWS) \
			.groupby(["source_user"]):
		if len(group.index) > MIN_GROUP_SIZE:
			user_name = group.iloc[0].get('source_user').split('@')[0]

			if user_name == "ANONYMOUS_LOGON":
				continue

			training_set, test_set = split_dataset(group)
			user = {
				"user_name": user_name,
				"datasets": {
					"training": training_set,
					"test": test_set
				}
			}
			users_list.append(user)

	print("Done gathering data, outputting to file ", output_file)
	output = open(output_file, 'wb')
	pickle.dump(users_list, output)
	sys.exit()


if __name__ == "__main__":
	assert TEST_SET_PERCENTAGE + TRAINING_SET_PERCENTAGE == 100
	main(sys.argv[1:])
