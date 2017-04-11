import tensorflow as tf

"""The authentication type of the request"""
AUTH_TYPE = {
	'NLTM': 0,
	'NEGOTIATE': 1,
	'KERBEROS': 2,
	'MICROSOFT_AUTHENTICATION_PACKAGE_V1_0': 3,
	'MICROSOFT_AUTHENTICATION_PAC': 4,
	'MICROSOFT_AUTHENTICATION_PACKAGE': 5,
	'MICROSOFT_AUTHENTICATION_PACKAG': 6,
	'MICROSOFT_AUTHENTICATION_PACKAGE_V1': 7,
	'MICROSOFT_AUTHENTICATION_PACKAGE_': 8,
	'MICROSOFT_AUTHENTICATION_PA': 9,
	'NONE': 10
}

"""The logon type of the request"""
LOGON_TYPE = {
	'NETWORK': 0,
	'SERVICE': 1,
	'BATCH': 2,
	'INTERACTIVE': 3,
	'NETWORKCLEARTEXT': 4,
	'NEWCREDENTIALS': 5,
	'UNLOCK': 6,
	'REMOTEINTERACTIVE': 7,
	'NONE': 8
}

"""The authentication orientation of the request"""
AUTH_ORIENTATION = {
	'LOGON': 0,
	'LOGOFF': 1,
	'TGS': 2,
	'AUTHMAP': 3,
	'TGT': 4
}

"""Whether the request was a success or failed"""
SUCCESS_FAILURE = {
	'SUCCESS': 0,
	'FAIL': 1
}

def str_to_enum(string, enum):
	"""Converts a string to a number in given enum"""
	upper_case = string.upper()

	if string == "?":
		return enum['NONE']

	return enum.get(upper_case)

def extract(row, model):
	"""Extracts the features for given row"""

	domains_amount = len(model.features.domains)
	dest_users_amount = len(model.features.dest_users)
	src_computers_amount = len(model.features.src_computers)
	dest_computers_amount = len(model.features.dest_computers)
	time_since_last_access = (row.time.to_pydatetime() - model.features.last_access).total_seconds()

	auth_type = str_to_enum(row.auth_type, AUTH_TYPE)
	logon_type = str_to_enum(row.logon_type, LOGON_TYPE)
	auth_orientation = str_to_enum(row.auth_orientation, AUTH_ORIENTATION)
	success_failure = str_to_enum(row.status, SUCCESS_FAILURE)


	feature_arr = [time_since_last_access, domains_amount, dest_users_amount,
				src_computers_amount, dest_computers_amount, auth_type,
				logon_type, auth_orientation, success_failure]
	print(feature_arr)
	return tf.convert_to_tensor([tf.to_float(feature_arr)], dtype=tf.float32, name="Features")