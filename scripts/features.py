"""The authentication type of the request"""
from typing import Dict, List


def zeros(length: int) -> List[float]:
    return [0.0] * length


def vectorize(values: Dict[str, int]) -> Dict[str, List[float]]:
    length = len(values)
    result = dict()
    for key, value in values.items():
        val_arr = zeros(length)
        val_arr[value - 1] = 1.0
        result[key] = val_arr

    return result


class Enum:
    def __init__(self, values: Dict[str, int]):
        self._values = vectorize(values)
        self._none_value = self._values.get('NONE')

    def get(self, string: str) -> List[float]:
        if string == '?':
            return self._none_value

        upper_case = string.upper()
        return self._values.get(upper_case) or self._none_value

    @property
    def length(self):
        return len(self._values)


ENUMS_LENGTH = 0
NON_ENUM_FEATURES_LENGTH = 7


def register_enum(values: Dict[str, int]) -> Enum:
    global ENUMS_LENGTH
    ENUMS_LENGTH += len(values)

    return Enum(values)


"""The authentication type used"""
AUTH_TYPE = register_enum({
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
})

"""The logon type of the request"""
LOGON_TYPE = register_enum({
    'NETWORK': 0,
    'SERVICE': 1,
    'BATCH': 2,
    'INTERACTIVE': 3,
    'NETWORKCLEARTEXT': 4,
    'NEWCREDENTIALS': 5,
    'UNLOCK': 6,
    'REMOTEINTERACTIVE': 7,
    'NONE': 8
})

"""The authentication orientation of the request"""
AUTH_ORIENTATION = register_enum({
    'LOGON': 0,
    'LOGOFF': 1,
    'TGS': 2,
    'AUTHMAP': 3,
    'TGT': 4,
    'NONE': 5
})


def get_delta(new: int, old: int) -> int:
    if new > old:
        return 1
    return 0


def extract(row, features, last_features) -> List[float]:
    """Extracts the features for given row"""

    domains_delta = get_delta(features.domains.unique, last_features.domains.unique)
    dest_users_delta = get_delta(features.dest_users.unique, last_features.dest_users.unique)
    src_computers_delta = get_delta(features.src_computers.unique, last_features.src_computers.unique)
    dest_computers_delta = get_delta(features.dest_computers.unique, last_features.dest_computers.unique)

    time_since_last_access = features.get_time_since_last_access()

    auth_type = AUTH_TYPE.get(row.auth_type)
    logon_type = LOGON_TYPE.get(row.logon_type)
    auth_orientation = AUTH_ORIENTATION.get(row.auth_orientation)
    success_failure = 0 if row.status == 'SUCCESS' else 1

    percentage_failed_logins = features.percentage_failed_logins

    non_enum_features = [time_since_last_access, domains_delta, dest_users_delta, src_computers_delta,
                         dest_computers_delta, percentage_failed_logins, success_failure]
    enums = auth_type + logon_type + auth_orientation

    assert len(non_enum_features) == NON_ENUM_FEATURES_LENGTH, "Non enum features have specified length"
    assert len(enums) == ENUMS_LENGTH, "Enum features have the specified length"
    feature_arr = non_enum_features + enums
    return feature_arr


def size():
    return NON_ENUM_FEATURES_LENGTH + ENUMS_LENGTH
