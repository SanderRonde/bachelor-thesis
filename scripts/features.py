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

    return enum.get(upper_case) or 0

def extract(row, features):
    """Extracts the features for given row"""

    unique_domains = features.domains.unique
    unique_dest_users = features.dest_users.unique
    unique_src_computers = features.src_computers.unique
    unique_dest_computers = features.dest_computers.unique

    most_freq_src_computer = features.src_computers.freq
    most_freq_dest_computer = features.dest_computers.freq
    time_since_last_access = features.get_time_since_last_access()

    auth_type = str_to_enum(row.auth_type, AUTH_TYPE)
    logon_type = str_to_enum(row.logon_type, LOGON_TYPE)
    auth_orientation = str_to_enum(row.auth_orientation, AUTH_ORIENTATION)
    success_failure = str_to_enum(row.status, SUCCESS_FAILURE)

    percentage_failed_logins = features.percentage_failed_logins


    feature_arr = [time_since_last_access, unique_domains, unique_dest_users,
                   unique_src_computers, unique_dest_computers, most_freq_src_computer,
                   most_freq_dest_computer, auth_type, logon_type, auth_orientation,
                   success_failure, percentage_failed_logins]
    return feature_arr
