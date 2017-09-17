import os
import time
import json
import sys
import math
import numpy as np
import pandas as pd
from typing import Union, List, Dict
from imports.timer import Timer
from imports.log import logline, debug, error
from imports.io import IO, IOInput

DATASET_ROWS = {
    'auth': 1051430459
}

REPORT_SIZE = 100
REMOVE_INPUT_FILE = False

# True = take the first x% of the data regardless of users
# False = take the first x% of users regardless of the amount of actions
DO_ROWS_PERCENTAGE = False


io = IO({
    'i': IOInput('/data/s1495674/anomalies.encoded.json', str, arg_name='input_file',
                 descr='The source file for the users (in json format)',
                 alias='input_file'),
    'o': IOInput('/data/s1495674/anomalies.json', str, arg_name='output_file',
                 descr='The file to output the anomalies to (specifying stdout outputs to stdout)',
                 alias='output_file'),
    'd': IOInput('/data/s1481096/LosAlamos/data/auth_small.h5', str, arg_name='dataset_file',
                 descr='The dataset file to use (in h5 format)',
                 alias='dataset_file'),
    's': IOInput(None, str, arg_name='state_file',
                 descr='The state file for which to wait before starting',
                 alias='state_file'),
    'p': IOInput(5.0, float, arg_name='dataset_percentage',
                 descr='The percentage of the dataset to use',
                 alias='dataset_percentage'),
    'u': IOInput(False, bool, arg_name='users_only',
                 descr='Only use actual users, not computer users',
                 alias='users_only', has_input=False)
})


UserAnomalies = List[Dict[str, Union[int, List[float]]]]


class AnomalySource:
    def __init__(self, data: Union[Dict[str, UserAnomalies], List[Dict[str, UserAnomalies]]]):
        if type(data) != list:
            data = [data]
        self.data = data

    def get(self, user: str) -> Union[None, UserAnomalies]:
        for i in range(len(self.data)):
            if user in self.data[i]:
                return self.data[i][user]

        return None

    def get_users(self) -> List[str]:
        users = list()
        for i in range(len(self.data)):
            for user in self.data[i].keys():
                users.append(user)

        return users


def read_anomalies(input_file: str) -> AnomalySource:
    # Read JSON file
    with open(input_file, 'r') as in_file:
        return AnomalySource(json.loads(in_file.read()))


def group_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(df['source_user'].map(lambda source_user: source_user.split('@')[0]), sort=False)


def filter_users(f: pd.DataFrame) -> pd.DataFrame:
    logline('Generating anonymous users filter')
    anonymous_users_filter = ~(f['source_user'].str.contains('ANONYMOUS') & f['source_user'].str.contains('LOGON'))

    if io.get('users_only'):
        debug('Skipping all computer users')
        logline('Generating computer users filter')
        computer_users_filter = ~(f['source_user'].str.startswith('C') & f['source_user'].str.endswith('$'))

        full_filter = anonymous_users_filter & computer_users_filter
    else:
        full_filter = anonymous_users_filter

    logline('Applying filters')
    return f[full_filter]


def translate_feature_arr(feature_arr: List[float]) -> Dict[str, float]:
    return {
        "time_since_last_access": feature_arr[0],
        "unique_domains": feature_arr[1],
        "unique_dest_users": feature_arr[2],
        "unique_src_computers": feature_arr[3],
        "unique_dest_computers": feature_arr[4],
        "most_freq_src_computer": feature_arr[5],
        "most_freq_dest_computer": feature_arr[6],
        "percentage_failed_logins": feature_arr[7],
        "success_failure": feature_arr[8],
        "auth_type": feature_arr[9],
        "logon_type": feature_arr[10],
        "auth_orientation": feature_arr[11]
    }


def listify_df(df) -> List[Dict[str, Union[str]]]:
    df_list = json.loads(df.to_json(orient='records'))
    index = 0
    for row in df.itertuples():
        df_list[index]["time"] = str(row[0])
        index = index + 1
    return df_list


def get_state(state_file_location: str) -> int:
    try:
        with open(state_file_location, 'r+') as state_file:
            try:
                state_obj = json.loads(state_file.read())
            except Exception:
                error('State file does not exist, cancelling')
                sys.exit(2)

            if state_obj["error"]:
                error('An error occurred in another instance, exiting with error code', state_obj['error_code'])
                sys.exit(state_obj['error_code'])
            return state_obj['state']
    except FileNotFoundError:
        error('State file does not exist, cancelling')
        sys.exit(2)


def get_dataset_name():
    return io.get('dataset_file').split('/')[-1].split('.')[0]


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


def main():
    if not io.run:
        return

    state_file = io.get('state_file')
    input_file = io.get('input_file')
    output_file = io.get('output_file')
    dataset_file = io.get('dataset_file')

    logline('Loading dataset file...')
    f = pd.read_hdf(dataset_file, get_dataset_name(), start=0, stop=calc_rows_amount())
    logline('Filtering users')
    f = filter_users(f)
    logline('Grouping users')
    f = group_df(f)

    if state_file is not None:
        initial_state = get_state(state_file)
        logline('Waiting for state to reach different value, currently at ' + str(initial_state) + '...')
        while get_state(state_file) == initial_state:
            time.sleep(60)

        logline('State file has switched to ' + str(get_state(state_file)) + ', continuing execution')

    logline('Loading anomalies')
    anomalies = read_anomalies(input_file)

    anomaly_rows_list = dict()

    max_users = users
    if DO_ROWS_PERCENTAGE:
        max_users = math.ceil(users * 0.01 * io.get('dataset_percentage'))

    timer = Timer(math.ceil(len(f) * 0.01 * io.get('dataset_percentage')))

    for name, group in f:
        user_name = group.iloc[0].get('source_user').split('@')[0]

        anomaly_collection = anomalies.get(user_name)
        if anomaly_collection is not None:
            # Print those rows

            user_anomalies = list()
            for anomaly in anomaly_collection:
                anomaly_dict = {
                    "start": anomaly["start"],
                    "end": anomaly["end"],
                    "lines": listify_df(group.iloc[anomaly["start"]:anomaly["end"]]),
                    "final_features": translate_feature_arr(anomaly["final_row_features"]),
                    "predicted": anomaly["predicted"],
                    "actual": anomaly["actual"],
                    "loss": anomaly["loss"]
                }
                user_anomalies.append(anomaly_dict)

            anomaly_rows_list[user_name] = user_anomalies

            timer.add_to_current(1)

        if timer.current % REPORT_SIZE == 0:
            logline('ETA is ' + timer.get_eta())

        if timer.current >= max_users:
            break

    debug('Runtime is', timer.report_total_time())
    logline('Generating concatenated results')
    if output_file == 'stdout':
        logline("Outputting results to stdout\n\n\n")
        logline('Final value is', anomaly_rows_list)
        logline(json.dumps(anomaly_rows_list))
    else:
        logline('Outputting results to', output_file)
        with open(output_file, 'w') as out_file:
            out_file.write(json.dumps(anomaly_rows_list))
            logline('Output results to', output_file)

    if REMOVE_INPUT_FILE:
        os.remove(input_file)
        logline('Removed encoded file')
    else:
        logline('Not Removing encoded file')

    logline('Done, closing files and stuff')


if __name__ == '__main__':
    start_time = time.time()
    main()
    logline('Total runtime is', Timer.stringify_time(Timer.format_time(time.time() - start_time)))
