import os
import sys
import json
import getopt
import pandas as pd
from typing import Tuple, Union, List, Dict

MAX_ROWS = None


def get_io_settings(argv: sys.argv) -> Tuple[str, str, str]:
    """This gets the input and output files from the command line arguments"""
    input_file = '/data/s1495674/anomalies.encoded.json'
    dataset_file = '/data/s1481096/LosAlamos/data/auth_small.h5'
    output_file = '/data/s1495674/anomalies.txt'

    try:
        opts, args = getopt.getopt(argv, 'i:o:d:')
    except getopt.GetoptError:
        print("Command line arguments invalid")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-i':
            input_file = arg
        elif opt == '-o':
            output_file = arg
        elif opt == '-d':
            dataset_file = arg
        elif opt == '-h':
            print("Options:")
            print(" -i <input file>     The source file for the users (in pickle format)")
            print(" -o <output file>    The file to output the anomalies to, specifying 'stdout' prints them to stdout")
            print(' -d <dataset_file>   The dataset file to use')
            sys.exit()
        else:
            print("Unrecognized argument passed, refer to -h for help")
            sys.exit(2)

    return input_file, output_file, dataset_file


UserAnomalies = List[Dict[str, float]]


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


def main():
    input_file, output_file, dataset_file = get_io_settings(sys.argv[1:])

    anomalies = read_anomalies(input_file)

    f = pd.read_hdf(dataset_file, 'auth_small', start=0, stop=MAX_ROWS) \
        .groupby(['source_user'])

    anomaly_rows_list = list()

    for name, group in f:
        user_name = group.iloc[0].get('source_user').split('@')[0]
        if user_name == "ANONYMOUS LOGON" or user_name == "ANONYMOUS_LOGON":
            continue

        anomaly_collection = anomalies.get(user_name)
        if anomaly_collection is not None:
            # Print those rows

            for anomaly in anomaly_collection:
                anomaly_rows_list.append(group.iloc[anomaly["start"]:anomaly["end"]])

    if output_file == 'stdout':
        print("Outputting results to stdout\n\n\n")
        print(pd.concat(anomaly_rows_list).to_csv(index=False, header=False))
    else:
        print('Outputting results to', output_file)
        with open(output_file, 'w') as output_file:
            output_file.write(pd.concat(anomaly_rows_list).to_csv(index=False, header=False))

    print('Removing encoded file')
    os.remove(input_file)

    print('Done, closing files and stuff')


if __name__ == '__main__':
    main()