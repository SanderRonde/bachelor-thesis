from enum import Enum

class Auth_type(Enum):
	"""The authentication type of the request"""

	NLTM = 0
	NEGOTIATE = 1
	KERBEROS = 2
	MICROSOFT_AUTHENTICATION_PACKAGE_V1_0 = 3
	MICROSOFT_AUTHENTICATION_PAC = 4
	MICROSOFT_AUTHENTICATION_PACKAGE = 5
	MICROSOFT_AUTHENTICATION_PACKAG = 6
	MICROSOFT_AUTHENTICATION_PACKAGE_V1 = 7
	MICROSOFT_AUTHENTICATION_PACKAGE_ = 8
	MICROSOFT_AUTHENTICATION_PA = 9
	NONE = 10

	def match(self, value):
		upper_case = value.upper()

		for name, member in self.__members__.items():
			if name == value:
				return member
		return self.NONE

class Logon_type(Enum):
	"""The logon type of the request"""

	NETWORK = 0
	SERVICE = 1
	BATCH = 2
	INTERACTIVE = 3
	NETWORKCLEARTEXT = 4
	NEWCREDENTIALS = 5
	UNLOCK = 6
	REMOTEINTERACTIVE = 7
	NONE = 8

	def match(self, value):
		upper_case = value.upper()

		for name, member in self.__members__.items():
			if name == value:
				return member
		return self.NONE

class Auth_orientation(Enum):
	"""The authentication orientation of the request"""
	
	LOGON = 0
	LOGOFF = 1
	TGS = 2
	AUTHMAP = 3
	TGT = 4

	def match(self, value):
		upper_case = value.upper()

		for name, member in self.__members__.items():
			if name == value:
				return member
		return self.NONE

class Success_failure(Enum):
	"""Whether the request was a success or failed"""

	SUCCESS = 0
	FAIL = 1

	def match(self, value):
		upper_case = value.upper()

		for name, member in self.__members__.items():
			if name == value:
				return member
		return self.NONE

def str_to_enum(string, enum):
	"""Converts a string to a number in given enum"""
	return enum.match(string).value


def extract(row, model):
	"""Extracts the features for given row"""

	domains_amount = len(model.features.domains)
	dest_users_amount = len(model.features.dest_users)
	src_computers_amount = len(model.features.src_computers)
	dest_computers_amount = len(model.features.dest_computers)
	time_since_last_access =  row.time - model.features.last_access

	auth_type = str_to_enum(row.auth_type, Auth_type)
	logon_type = str_to_enum(row.logon_type, Logon_type)
	auth_orientation = str_to_enum(row.auth_orientation, Auth_orientation)
	success_failure = str_to_enum(row.status, Success_failure)

	return tf.convert_to_tensor(list(time_since_last_access, domains_amount, dest_users_amount,
				src_computers_amount, dest_computers_amount, auth_type,
				logon_type, auth_orientation, success_failure), name="Features")