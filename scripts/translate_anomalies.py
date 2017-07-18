import os
import json
import numpy as np
import pandas as pd
from typing import Union, List, Dict
from imports.timer import Timer
from imports.log import logline
from imports.io import IO, IOInput

MAX_ROWS = None
REPORT_SIZE = 100
TRAINING_SET_PERCENTAGE = 70


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
    'M': IOInput(False, bool, has_input=False,
                 descr='Enable meganet mode',
                 alias='meganet')
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


def read_anomalies(input_file: str) -> AnomalySource:
    # Read JSON file
    with open(input_file, 'r') as in_file:
        return AnomalySource(json.loads(in_file.read()))


def group_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(df['source_user'].map(lambda source_user: source_user.split('@')[0]), sort=False)


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


def main():
    if not io.run:
        return

    input_file = io.get('input_file')
    output_file = io.get('output_file')
    dataset_file = io.get('dataset_file')

    anomalies = read_anomalies(input_file)

    f = pd.read_hdf(dataset_file, 'auth', start=0, stop=MAX_ROWS)
    if io.get('meganet'):
        test_set = list()
        index = 0
        for g, dataframe in f.groupby(np.arange(10)):
            if index * 10 > TRAINING_SET_PERCENTAGE:
                test_set.append(dataframe)
        # noinspection PyTypeChecker
        f = group_df(pd.concat(test_set))
    else:
        f = group_df(f)

    anomaly_rows_list = dict()

    timer = Timer(len(f))

    for name, group in f:
        user_name = group.iloc[0].get('source_user').split('@')[0]
        if user_name == "ANONYMOUS LOGON" or user_name == "ANONYMOUS_LOGON":
            continue

        anomaly_collection = anomalies.get(user_name)
        if anomaly_collection is not None:
            # Print those rows

            user_anomalies = list()
            for anomaly in anomaly_collection:
                user_anomalies.append({
                    "start": anomaly["start"],
                    "end": anomaly["end"],
                    "lines": group.iloc[anomaly["start"]:anomaly["end"]],
                    "final_features": translate_feature_arr(anomaly["final_row_features"])
                })

            anomaly_rows_list[user_name] = user_anomalies

            timer.add_to_current(1)

        if timer.current % REPORT_SIZE == 0:
            logline('ETA is ' + timer.get_eta())

    logline('Generating concatenated results')
    if output_file == 'stdout':
        logline("Outputting results to stdout\n\n\n")
        logline(json.dumps(combined_values))
    else:
        logline('Outputting results to', output_file)
        with open(output_file, 'w') as output_file:
            output_file.write(json.dumps(combined_values))

    logline('Not Removing encoded file')
    #os.remove(input_file)

    logline('Done, closing files and stuff')


if __name__ == '__main__':
    main()
