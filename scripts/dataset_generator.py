"""The script that generates all 3 datasets and their features"""
import math
import pickle
import pytz
import sys
import json
import time
from imports.timer import Timer
from imports.log import logline, debug, error
from imports.io import IO, IOInput
from typing import List, TypeVar, Tuple, Union, Dict, Any
import traceback

import features as features
import numpy as np
import pandas as pd
import multiprocessing

T = TypeVar('T')

DATASET_ROWS = {
    'auth': 1051430459
}

TRAINING_SET_PERCENTAGE = 70
REPORT_SIZE = 1000
BATCH_SIZE = 32
MIN_GROUP_SIZE = 150
MIN_GROUP_SIZE = max(MIN_GROUP_SIZE, (BATCH_SIZE * 2) + 2)
PROCESSING_GROUP_SIZE = 500
SKIP_MAIN = False

# True = take the first x% of the data regardless of users
# False = take the first x% of users regardless of the amount of actions
DO_ROWS_PERCENTAGE = False

io = IO({
    'i': IOInput('/data/s1481096/LosAlamos/data/auth.h5', str, arg_name='input_file',
                 descr='The source file for the users (in h5 format)',
                 alias='input_file'),
    'o': IOInput('/data/s1495674/features.p', str, arg_name='output_file',
                 descr='The file to output the features to',
                 alias='output_file'),
    'c': IOInput(1, int, arg_name='cpus',
                 descr='The number of CPUs to use',
                 alias='cpus'),
    'n': IOInput('auth', str, arg_name='dataset_name',
                 descr='The name of the pandas object in the dataset file',
                 alias='dataset_name'),
    'p': IOInput(100.0, float, arg_name='dataset_percentage',
                 descr='The percentage of the amount of users to use',
                 alias='dataset_percentage'),
    'u': IOInput(False, bool, arg_name='users_only',
                 descr='Only use actual users, not computer users',
                 alias='users_only', has_input=False)
})


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
    def __init__(self, _list: list = None, _last: str = None):
        self._list = _list or list()
        self._last = _last

    def append(self, item: str):
        """Appends given item to the list of the property"""
        if item not in self._list:
            self._list.append(item)
        self._last = item

    @property
    def unique(self) -> int:
        # Get the length of the unique items of the last X items
        return len(self._list)

    @property
    def last(self) -> int:
        if self._last in self._list:
            return self._list.index(self._last)
        return 0

    @property
    def list(self) -> List[str]:
        return self._list

    def snapshot(self):
        return PropertyDescription(_list=self._list[:], _last=self._last)


class Features:
    """All the features fr a model"""

    def __init__(self, current_access: int = 0,
                 last_access: int = 0,
                 domains = None,
                 dest_users = None,
                 src_computers = None,
                 dest_computers = None,
                 failed_logins: int = 0,
                 login_attempts: int = 0):
        self._current_access = current_access
        self._last_access = last_access
        self._domains = domains or PropertyDescription()
        self._dest_users = dest_users or PropertyDescription()
        self._src_computers = src_computers or PropertyDescription()
        self._dest_computers = dest_computers or PropertyDescription()
        self._failed_logins = failed_logins
        self._login_attempts = login_attempts

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

    def snapshot(self):
        return Features(
            current_access=self.current_access,
            last_access=self._last_access,
            domains=self._domains.snapshot(),
            dest_users=self._dest_users.snapshot(),
            src_computers=self._src_computers.snapshot(),
            dest_computers=self._dest_computers.snapshot(),
            failed_logins=self._failed_logins,
            login_attempts=self._login_attempts
        )


def normalize_all(feature_list: List[List[float]]) -> Tuple[List[float], np.ndarray]:
    copy = feature_list[:]
    np_arr = np.array(copy).astype(float)
    swapped = np.swapaxes(np_arr, 0, 1)

    scales = list()
    for row in range(len(swapped)):
        row_max = max(swapped[row])
        if row_max == 0.0:
            scales.append(0.0)
            continue
        scales.append(row_max)
        swapped[row] = [float(i) / row_max for i in swapped[row]]

    return scales, np.swapaxes(swapped, 0, 1)


def convert_to_features(data_part) -> np.ndarray:
    """This converts a given group to features"""
    current_features = Features()

    feature_list = list()
    last_features = current_features
    for row in data_part.itertuples():
        row_data = Row(row)
        current_features.update(row_data)
        feature_list.append(features.extract(row_data, current_features, last_features))
        last_features = current_features.snapshot()

    return np.array(feature_list)


def closest_multiple(target: int, base: int) -> int:
    lower_bound = (target // base) * base
    if float(target - lower_bound) > (base / 2):
        # Round up
        return lower_bound + base
    return lower_bound


def split_list(target: np.ndarray, batch_size: int = 1) -> Union[Tuple[np.ndarray, np.ndarray], None]:
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
        return None

    return target[0:training_set_length], target[training_set_length:training_set_length + test_set_length]


def split_dataset(feature_data: np.ndarray) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
    """This converts the dataset to features and splits it into 3 parts"""
    result = split_list(feature_data, BATCH_SIZE)
    if result:
        return result[0], result[1]
    return None, None


def get_dataset_name() -> str:
    return io.get('dataset_name') or io.get('input_file').split('/')[-1].split('.')[0]


def calc_rows_amount() -> Union[int, None]:
    dataset_name = get_dataset_name()

    if not DO_ROWS_PERCENTAGE:
        return None

    if dataset_name in DATASET_ROWS:
        all_rows = DATASET_ROWS.get(dataset_name)
    elif io.get('dataset_percentage') == 100.0:
        return None
    else:
        debug('Reading percentages of unknown datasets is not possible,'
              'please add the dataset name and amount of rows to the'
              'DATASET_ROWS variable in this file and try again')
        debug('Using all rows instead')
        return None

    return round((all_rows / 100) * io.get('dataset_percentage'))


def get_pd_file() -> pd.DataFrame:
    logline('Opening file')
    dataset_name = get_dataset_name()

    return pd.read_hdf(io.get('input_file'), dataset_name, start=0, stop=calc_rows_amount())


def group_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(df['source_user'].map(lambda source_user: source_user.split('@')[0]), sort=False)


def group_pd_file(f: pd.DataFrame) -> pd.DataFrame:
    logline('Grouping users in file')
    grouped = group_df(f)
    logline('Done grouping users')
    return grouped


def split_dataframe(f: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Try to get close to the target split
    training_set = list()
    test_set = list()

    index = 10
    logline('Splitting dataframes')
    grouped = f.groupby(np.arange(len(f)) // (len(f) / 10))
    for g, dataframe in grouped:
        if index <= TRAINING_SET_PERCENTAGE:
            training_set.append(dataframe)
        else:
            test_set.append(dataframe)
        index += 10

    # noinspection PyTypeChecker
    return pd.concat(training_set), pd.concat(test_set)


def get_lower_bound(maximum: int, base: int) -> int:
    return (maximum // base) * base


class DatasetContainer:
    def __init__(self, group_len: int, user_name: str = None, training_set: np.ndarray = None,
                 test_set: np.ndarray = None, error: bool = False):
        self.user_name = user_name
        self.training_set = training_set
        self.test_set =test_set
        self.group_len = group_len
        self.error = error

    def to_dict(self) -> Dict[str, Union[str, int, Dict[str, np.ndarray]]]:
        if self.error:
            return {
                "error": True,
                "group_len": self.group_len
            }
        return {
            "user_name": self.user_name,
            "datasets": {
                "training": self.training_set,
                "test": self.test_set
            },
            "group_len": self.group_len
        }


def gen_features_for_user(args: Tuple[str, Any]) -> Dict[str, Union[str, int, Dict[str, np.ndarray]]]:
    name = args[0]
    group = args[1]
    empty_result = {
        "error": True,
        "group_len": len(group)
    }
    if len(group.index) > MIN_GROUP_SIZE:
        # debug('Doing group with length', len(group))
        user_name = name

        if user_name == "ANONYMOUS LOGON" or user_name == "ANONYMOUS_LOGON":
            return empty_result
        if io.get('users_only') and user_name.startswith('C') and user_name.endswith('$'):
            return empty_result

        scales, normalized = normalize_all(convert_to_features(group))
        split_dataset_result = split_dataset(normalized)
        if split_dataset_result:
            training_set, test_set = split_dataset_result
            return {
                "user_name": user_name,
                "datasets": {
                    "training": training_set,
                    "test": test_set,
                    "scales": scales
                },
                "group_len": len(group)
            }
    else:
        return empty_result


class DFIterator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.index = 0
        self.max = 0

        self.df_iterator = df.__iter__()

    def set_max(self, maximum: int):
        self.max = maximum

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.max:
            raise StopIteration
        else:
            self.index += 1
            return self.df_iterator.__next__()


def strip_group_length(data) -> Tuple[Union[Dict[str, Any], None], int]:
    if 'error' in data and data['error']:
        if data['group_len'] == -1:
            error('Value too big for pickle returned, skipping it, ETA might be off now')
            return None, 0
        return None, data['group_len']

    group_length = data['group_len']
    return {
        "user_name": data['user_name'],
        "datasets": data['datasets']
    }, group_length


def gen_features(f: pd.DataFrame, row_amount: int) -> List[Dict[str, Union[str, Dict[str, List[List[float]]]]]]:
    users_list = list()

    logline('Calculating amount of groups...')
    users = len(f)
    logline('There are', users, 'users and', row_amount, 'rows')
    rows = 0

    max_users = users
    if DO_ROWS_PERCENTAGE:
        max_users = int(math.ceil(users * 0.01 * io.get('dataset_percentage')))
    logline('Max amount of usersis', max_users)

    logline('Setting timer for', int(math.ceil(row_amount * 0.01 * io.get('dataset_percentage'))), 'users')
    timer = Timer(int(math.ceil(row_amount * 0.01 * io.get('dataset_percentage'))))

    logline('Creating iterator')
    dataset_iterator = DFIterator(f)

    if not SKIP_MAIN:
        try:
            # Create groups of approx 1000 users big
            if io.get('cpus') == 1:
                logline('Only using a single CPU')
                logline('Starting feature generation')
                if io.get('users_only'):
                    debug('Skipping all computer users')
                for name, group in f:
                    completed_result, group_len = strip_group_length(gen_features_for_user((name, group)))
                    timer.add_to_current(group_len)
                    rows += group_len

                    if completed_result is not None:
                        users_list.append(completed_result)

                        if rows % REPORT_SIZE == 0:
                            logline('At row ', str(rows), '/~', str(row_amount), ' - ETA is: ' + timer.get_eta(),
                                    spaces_between=False)
                            logline('At user', len(users_list), '/~', max_users, spaces_between=False)

                    if len(users_list) >= max_users:
                        break

            else:
                logline('Using', io.get('cpus'), 'cpus')
                for i in range(round(math.ceil(max_users / PROCESSING_GROUP_SIZE))):
                    dataset_iterator.set_max((i + 1) * PROCESSING_GROUP_SIZE)
                    if i == 0:
                        logline('Starting feature generation')

                    if io.get('users_only'):
                        debug('Skipping all computer users')
                    with multiprocessing.Pool(io.get('cpus')) as p:
                        for completed_result in p.imap_unordered(gen_features_for_user, dataset_iterator, chunksize=100):

                            completed_result, group_len = strip_group_length(completed_result)
                            timer.add_to_current(group_len)
                            rows += group_len

                            if completed_result is not None:
                                users_list.append(completed_result)

                                if rows % REPORT_SIZE == 0:
                                    logline('At row ', str(rows), '/~', str(row_amount), ' - ETA is: ' + timer.get_eta()
                                            , spaces_between=False)
                                    logline('At user', len(users_list), '/~', max_users, spaces_between=False)
        except KeyboardInterrupt:
            logline('User cancelled execution, wrapping up')
            debug('Cancelled early at', len(users_list), 'instead of', users)
            debug('You skipped a total of', users - len(users_list), 'users, or',
                  100 - ((len(users_list) / users) * 100), '%')
        except Exception:
            error('An error occurred during execution', traceback.format_exc())
            debug('Salvaging all remaining users')
        finally:
            debug('Runtime is', timer.report_total_time())

            logline("Did a total of", len(users_list), "users")
            logline('Done gathering data')
            logline('Closing file...')
            return users_list
    else:
        debug('SKIPPING MAIN, DO NOT ENABLE IN PRODUCTION')
        logline('Closing file')
        return []


def get_features() -> Union[Dict[str, List[Dict[str, Union[str, List[List[float]]]]]],
                            List[Dict[str, Union[str, Dict[str, List[List[float]]]]]]]:
    file = get_pd_file()
    row_amount = len(file)
    f = group_pd_file(file)
    return gen_features(f, row_amount)


def output_data(users_list: List[Dict[str, Union[str, Dict[str, List[List[float]]]]]]):
    if io.get('output_file') == 'stdout':
        logline('Outputting to stdout')
        sys.stdout.write(json.dumps(users_list))
    else:
        logline('Outputting data to file', io.get('output_file'))
        output = open(io.get('output_file'), 'wb')
        try:
            pickle.dump(users_list, output, protocol=4)
        except:
            try:
                logline("Using JSON instead")
                output.write(json.dumps(users_list))
            except:
                error('Outputting to console instead')
                print(json.dumps(users_list))
                raise
            raise


def main():
    if not io.run:
        return

    start_time = time.time()
    logline("Gathering features for", str(io.get('dataset_percentage')) + "% of rows",
            "using a batch size of", BATCH_SIZE)

    output_data(get_features())
    logline('Total runtime is', Timer.stringify_time(Timer.format_time(time.time() - start_time)))
    sys.exit()


if __name__ == "__main__":
    main()
