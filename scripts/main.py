"""The main script file"""
import os
import _pickle as pickle
import sys
import time
import json
import traceback
import tensorflow as tf
from typing import List, Dict, Tuple, Union, TypeVar, Any
import matplotlib.pyplot as plt
import numpy as np
import features
from imports.timer import Timer
from imports.log import logline_to_folder
from imports.io import IO, IOInput

io = IO({
    "i": IOInput('/data/s1495674/features.p', str, arg_name='input_file',
                 descr='The source file for the users (in pickle format)',
                 alias='input_file'),
    "o": IOInput('/data/s1495674/anomalies.encoded.json', str, arg_name='out_file',
                 descr='The file to output the anomalies to, specifying \'stdout\' prints them to stdout',
                 alias='output_file'),
    'v': IOInput(False, bool, has_input=False, descr="Enable verbose output mode", alias='verbose'),
    'k': IOInput(True, bool, has_input=False, descr="Specifying this disables keeping the training set in the state "
                                                    "before trying the test set", alias='give_prev_knowledge'),
    'e': IOInput(25, int, arg_name='epochs', descr="The amount of epochs to use (default is 25)",
                 alias='epochs'),
    's': IOInput(0, int, arg_name='percentage', descr='The index at which to start processing',
                 alias='start'),
    'd': IOInput(100, int, arg_name='percentage', descr='The index at which to stop processing',
                 alias='end'),
    'x': IOInput(True, bool, has_input=False, descr='Disable verbose output during running',
                 alias='running_verbose'),
    'p': IOInput('/data/s1495674/plot_data/', str, arg_name='plot_location',
                 descr="The location to store the plotting data",
                 alias='plot_location'),
    'l': IOInput(None, str, arg_name='log_folder',
                 descr='The to output the logs',
                 alias='log_folder')
})

# Constants
BATCH_SIZE = 32
SPEED_REPORTING_SIZE = 1000
MAX_HIGHEST_OFFENDERS = 10
GROUP_LENGTH = 32

# Global variables/functions
EPOCHS = io.get('epochs')
GIVE_TEST_SET_PREVIOUS_KNOWLEDGE = io.get('give_prev_knowledge')
VERBOSE = io.get('verbose')
VERBOSE_RUNNING = io.get('running_verbose')
USES_DIFFERENT_INDEXES = io.get('start') != 0 or io.get('end') != 100
logline, debug, error, log_to_folder_done = logline_to_folder(folder_loc=io.get('log_folder'),
                                                              file_name='main_logs.log',
                                                              start=io.get('start'), end=io.get('end'))
T = TypeVar('T')

if io.run:
    from keras.layers.core import Dense
    from keras.layers.recurrent import LSTM
    from keras.models import Sequential

plt.switch_backend('agg')

np.random.seed(1234)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PLOTS = {
    "LOSSES": dict(),
    "TIME_SINCE_LAST_ACCESS": dict(),
    "UNIQUE_DOMAINS": dict(),
    "UNIQUE_DEST_USERS": dict(),
    "PERCENTAGE_FAILED_LOGINS": dict(),
    "IQRS": dict(),
    "IQR_MAXES": dict(),
    'DEVIATIONS': list(),
    "ALL_DEVIATIONS": list(),

    # Non-plot plots, just data gathering
    "USER_TIMINGS_TRAINING": list(),
    "USER_TIMINGS_TEST": list()
}
FEATURE_SET = List[List[float]]

TIME_SINCE_LAST_ACCESS_INDEX = 0
UNIQUE_DOMAINS_INDEX = 1
UNIQUE_DEST_USERS_INDEX = 2
PERCENTAGE_FAILED_LOGINS_INDEX = 7


def force_str(val: str) -> str:
    return val


def force_feature(val: List[float]) -> List[float]:
    return val


def force_feature_set(val: FEATURE_SET) -> FEATURE_SET:
    return val


def force_list_of_int(val: List[int]) -> List[int]:
    return val


class DataDistribution:
    def __init__(self, data: Dict[str, FEATURE_SET], user_name: str):
        self.training = force_feature_set(data["training"])
        self.test = force_feature_set(data["test"])

        PLOTS["TIME_SINCE_LAST_ACCESS"][user_name] = {
            "max": max(
                self.training[-1][TIME_SINCE_LAST_ACCESS_INDEX],
                self.test[-1][TIME_SINCE_LAST_ACCESS_INDEX]),
            "all": list(np.array(self.test)[:, TIME_SINCE_LAST_ACCESS_INDEX])
        }
        PLOTS["UNIQUE_DOMAINS"][user_name] = {
            "max": max(
                self.training[-1][UNIQUE_DOMAINS_INDEX],
                self.test[-1][UNIQUE_DOMAINS_INDEX]),
            "all": list(np.array(self.test)[:, UNIQUE_DOMAINS_INDEX])
        }
        PLOTS["UNIQUE_DEST_USERS"][user_name] = {
            "max": max(
                self.training[-1][UNIQUE_DEST_USERS_INDEX],
                self.test[-1][UNIQUE_DEST_USERS_INDEX]),
            "all": list(np.array(self.test)[:, UNIQUE_DEST_USERS_INDEX])
        }
        PLOTS["PERCENTAGE_FAILED_LOGINS"][user_name] = {
            "max": max(
                self.training[-1][PERCENTAGE_FAILED_LOGINS_INDEX],
                self.test[-1][PERCENTAGE_FAILED_LOGINS_INDEX]),
            "all": list(np.array(self.test)[:, PERCENTAGE_FAILED_LOGINS_INDEX])
        }


class Dataset:
    def __init__(self, data: Dict[str, Union[str, List[int], List[float], Dict[str, FEATURE_SET]]]):
        self.user_name = force_str(data["user_name"])
        self.datasets = DataDistribution(data["datasets"], self.user_name)


class FeatureDescriptor:
    def __init__(self, name: str, type_: str, weight: float):
        self.name = name
        self.type = type_
        self.weight = weight


FEATURE_SIZE = features.size()
LAYERS = [FEATURE_SIZE, 128, 128, FEATURE_SIZE]


def create_anomaly(start: int, end: int, train_len: int, dataset: FEATURE_SET) -> Dict[str, Union[int, List[float]]]:
    final_row = dataset[end - 1]
    return {
        "start": start + train_len,
        "end": end + train_len,
        "final_row_features": final_row
    }


class LossesGroup:
    def __init__(self, first_loss: float, is_anomaly: bool):
        self.losses = [first_loss]
        self.anomalies = 1 if is_anomaly else 0

    @property
    def length(self):
        return len(self.losses)

    def append(self, loss: float):
        self.losses.append(loss)

    def add_anomaly(self):
        self.anomalies += 1


class LossesGroupMetadata:
    def __init__(self, anomaly_score: float, start: int, end: int):
        self.anomaly_score = anomaly_score
        self.start = start
        self.end = end


class RNNModel:
    """An RNN"""

    def __init__(self, group_size=GROUP_LENGTH):
        model = Sequential()
        model.add(LSTM(LAYERS[1], input_shape=(FEATURE_SIZE, 1), batch_size=BATCH_SIZE,
                       return_sequences=True, stateful=True))
        model.add(LSTM(LAYERS[2], return_sequences=False, stateful=True))
        model.add(Dense(LAYERS[3]))
        model.compile(loss='mean_squared_error', optimizer='adam')

        self.model = model
        self.group_size = group_size

        self._starting_weights = list()
        for i in range(len(model.layers)):
            self._starting_weights.append(model.layers[i].get_weights())

    @staticmethod
    def prepare_data(training_data: FEATURE_SET, test_data: FEATURE_SET = None):
        """Prepares given datasets for insertion into the model"""

        if len(training_data) == 1:
            error('Training data is not big enough', training_data, test_data)
        assert len(training_data) > 1, "Training data is longer than 1, (is %d)" % len(training_data)
        if test_data is not None:
            assert len(test_data) > 1, "Test data is longer than 1, (is %d)" % len(test_data)

        train_x = np.array(training_data[:-1])
        train_y = np.array(training_data[1:])

        train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))

        if test_data is not None:
            test_x = np.array(test_data[:-1])
            test_y = np.array(test_data[1:])

            max_groups = len(test_x) + 1 - BATCH_SIZE

            x_groups = list()
            y_groups = list()

            for i in range(len(test_x)):
                if i < max_groups:
                    x_groups.append([test_x[i]])
                    y_groups.append([test_y[i]])

                for j in range(len(x_groups)):
                    if len(x_groups[j]) < BATCH_SIZE:
                        x_groups[j].append(test_x[i])
                        y_groups[j].append(test_y[i])

            # Serialize the groups
            test_x = list()
            test_y = list()

            for i in range(len(x_groups)):
                test_x = test_x + x_groups[i]
                test_y = test_y + y_groups[i]

            test_x = np.array(test_x)
            test_y = np.array(test_y)

            test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

            return train_x, train_y, test_x, test_y
        return train_x, train_y

    def reset(self, reset_weights=True):
        if reset_weights:
            for i in range(len(self.model.layers)):
                self.model.layers[i].set_weights(self._starting_weights[i])
        self.model.reset_states()

    def fit(self, train_x, train_y, epochs: int = 10):
        """Fits the model to given training data"""

        for i in range(epochs):
            global VERBOSE_RUNNING
            if VERBOSE_RUNNING:
                logline("Epoch", i, '/', epochs)
            verbosity = 0
            if VERBOSE_RUNNING:
                verbosity = 1
            self.model.fit(train_x, train_y, batch_size=BATCH_SIZE,
                           epochs=1, verbose=verbosity, shuffle=False)
            if not GIVE_TEST_SET_PREVIOUS_KNOWLEDGE or i != epochs - 1:
                self.model.reset_states()

    def test(self, test_x, test_y) -> List[LossesGroupMetadata]:
        """Predicts the result for given test data"""
        losses = list()

        assert len(test_x) % BATCH_SIZE == 0, 'Dataset should be divisible by batch size'
        for i in range(round(len(test_x) / BATCH_SIZE)):
            losses.append(self.model.evaluate(test_x[i*BATCH_SIZE:(i+1)*BATCH_SIZE],
                                              test_y[i*BATCH_SIZE:(i+1)*BATCH_SIZE],
                                              batch_size=BATCH_SIZE, verbose=0))

        groups = list()
        for i in range(len(losses)):
            groups.append(LossesGroupMetadata(losses[i], i, i + BATCH_SIZE))

        return groups


def abs_ratio(a: float, b: float) -> float:
    if b == 0:
        return 1000
    ratio = a / b
    if ratio < 1:
        return 1 / ratio
    return ratio


def iqr(data: List[float]) -> Tuple[float, float]:
    copy = data[:]
    copy.sort()

    # noinspection PyTypeChecker
    q3, q1 = np.percentile(data, [75, 25])
    inter_quartile_range = q3 - q1

    return inter_quartile_range, q3


def get_iqr_distance(inter_quartile_range: float, q3: float, value: float) -> float:
    # Solve:
    return (value - q3) / inter_quartile_range


class UserNetwork:
    """The class describing a single model and all its corresponding data"""

    def __init__(self, model: RNNModel, data: Dataset, epochs: int = 10):
        """Creates a new set of networks"""

        self.user_name = data.user_name
        self.dataset = data.datasets
        self.config = {
            "epochs": epochs
        }

        self.model = model
        self.model.reset()

    def get_losses(self, x: np.ndarray, y: np.ndarray) -> List[LossesGroupMetadata]:
        return self.model.test(x, y)

    def find_anomalies(self) -> List[Dict[str, Union[int, List[float]]]]:
        # Train the network first
        train_x, train_y, test_x, test_y = RNNModel.prepare_data(self.dataset.training, test_data=self.dataset.test)
        self.model.fit(train_x, train_y, epochs=self.config["epochs"])

        logline("\n")
        logline("Checking losses on test set...")
        test_losses = self.get_losses(test_x, test_y)
        logline("Done checking losses on test set\n")

        anomalies = list()

        only_losses = list(map(lambda x: x.anomaly_score, test_losses))

        # Interquartile range
        inter_quartile_range, q3 = iqr(only_losses)

        PLOTS["LOSSES"][self.user_name] = only_losses
        PLOTS["IQRS"][self.user_name] = inter_quartile_range
        PLOTS["IQR_MAXES"][self.user_name] = q3 + (1.5 * inter_quartile_range)

        for i in range(len(test_losses)):
            test_loss = test_losses[i]
            iqr_distance = get_iqr_distance(inter_quartile_range, q3, test_loss.anomaly_score)
            if iqr_distance > 1.5:
                anomaly = create_anomaly(test_loss.start, test_loss.end, len(train_x),
                                         test_y)
                anomalies.append(anomaly)

                PLOTS["DEVIATIONS"].append({
                    "key": self.user_name,
                    "val": iqr_distance
                })
            PLOTS["ALL_DEVIATIONS"].append(iqr_distance)

        return anomalies


def find_anomalies(model: RNNModel, data: Dataset) -> List[Dict[str, Union[int, List[float]]]]:
    """Finds anomalies in given data"""
    network = UserNetwork(model, data, epochs=EPOCHS)
    anomalies = network.find_anomalies()
    return anomalies


def save_plot(plot_location: str, name: str,
              data: Any,
              x_label: str, y_label: str,
              is_log: bool = False, normalize_x: bool = False, normalize_y: bool = False,
              multidimensional: bool = False, is_dict=False, is_box_plot=False, is_highest_offenders=False,
              is_plot: bool = True):
    plot_data = {
        "name": name,
        "x_label": x_label,
        "y_label": y_label,
        "is_log": is_log,
        "is_plot": is_plot,
        "normalize_x": normalize_x,
        "normalize_y": normalize_y,
        "plot_location": plot_location,
        "multidimensional": multidimensional,
        "is_dict": is_dict,
        "is_box_plot": is_box_plot,
        "is_highest_offenders": is_highest_offenders,
        "max_highest_offenders": MAX_HIGHEST_OFFENDERS,
        "data": data
    }

    if USES_DIFFERENT_INDEXES:
        plot_location = plot_location + name + '.part.' + str(io.get('start')) + '.' + str(io.get('end')) + '.json'
    logline("Outputting plot data to", plot_location)
    with open(plot_location, 'w') as out_file:
        out_file.write(json.dumps(plot_data))


def listifydict(data_dict: Dict[str, T], get_max=False, get_all=False) -> List[T]:
    data_list = list()
    for key, val in data_dict.items():
        if get_max:
            data_list.append(val["max"])
        elif get_all:
            data_list.append(val["all"])
        else:
            data_list.append(val)
    return data_list


def get_highest_vals(data_list: List[Dict[str, Union[str, float]]]) -> List[Dict[str, Union[str, float]]]:
    list_copy = data_list[:]
    list_copy.sort(key=lambda x: x["val"])

    for i in range(len(list_copy)):
        list_copy[i]["sorting_value"] = list_copy[i]["val"]
    return list_copy[-MAX_HIGHEST_OFFENDERS:]


def get_mean(data_list: List[Dict[str, Union[str, float]]]) -> float:
    total = 0.0
    for i in range(len(data_list)):
        total += data_list[i]["val"]

    if len(data_list) == 0:
        return 0.1
    return total / len(data_list)


def data_points_for_users(highest_offenders: List[Dict[str, Union[str, float]]],
                          data: Dict[str, float], key: str = None) -> Dict[str, Dict[str, float]]:
    match_dict = dict()
    for i in range(len(highest_offenders)):
        if key is None:
            match_dict[highest_offenders[i]["key"]] = {
                "val": data[highest_offenders[i]["key"]],
                "sorting_value": highest_offenders[i]["val"]
            }
        else:
            match_dict[highest_offenders[i]["key"]] = {
                "val": data[highest_offenders[i]["key"]][key],
                "sorting_value": highest_offenders[i]["val"]
            }

    return match_dict


def save_plot_data(plot_location: str):
    """Plots everything"""

    if not plot_location.endswith('/'):
        plot_location = plot_location + '/'

    logline('')
    logline('Finding highest offenders')
    highest_offenders = get_highest_vals(PLOTS["DEVIATIONS"])
    highest_offenders_with_mean_dict = {
        "avg.": get_mean(PLOTS["DEVIATIONS"])
    }
    for i in range(len(highest_offenders)):
        highest_offenders_with_mean_dict[highest_offenders[i]["key"]] = {
            "key": highest_offenders[i]["key"],
            "val": highest_offenders[i]["val"],
            "sorting_value": highest_offenders[i]["val"]
        }

    # Plots of the highest offenders
    logline('')
    logline('Plotting highest offenders plots')

    save_plot(plot_location, 'deviations', list(map(lambda x: x["val"], PLOTS["DEVIATIONS"])),
              'User index', 'Relative deviation (from mean)')
    save_plot(plot_location, 'highest_offender_deviations', highest_offenders_with_mean_dict,
              'User Name', 'Relative deviation (from mean)',
              is_dict=True, multidimensional=True, is_highest_offenders=True)
    save_plot(plot_location, 'highest_offender_time_since_last_access',
              data_points_for_users(highest_offenders, PLOTS["TIME_SINCE_LAST_ACCESS"], key="all"),
              'User Name', 'Max time since last access (in seconds)',
              is_dict=True, is_box_plot=True, is_highest_offenders=True)
    save_plot(plot_location, 'highest_offender_unique_domains',
              data_points_for_users(highest_offenders, PLOTS["UNIQUE_DOMAINS"], key="all"),
              'User Name', 'Max unique domains',
              is_dict=True, is_box_plot=True, is_highest_offenders=True)
    save_plot(plot_location, 'highest_offender_dest_users',
              data_points_for_users(highest_offenders, PLOTS["UNIQUE_DEST_USERS"], key="all"),
              'User Name', 'Max dest users',
              is_dict=True, is_box_plot=True, is_highest_offenders=True)
    save_plot(plot_location, 'highest_offender_failed_logins',
              data_points_for_users(highest_offenders, PLOTS["PERCENTAGE_FAILED_LOGINS"], key="all"),
              'User Name', 'Percentage failed logins',
              is_dict=True, is_box_plot=True, is_highest_offenders=True)

    # Plots of the losses of all batches
    logline('')
    logline('Plotting losses')
    save_plot(plot_location, 'all deviations', PLOTS["ALL_DEVIATIONS"],
              'Batch index', 'IQR Scale')
    save_plot(plot_location, 'losses', listifydict(PLOTS["LOSSES"]),
              'Batch index', 'Loss ratio',
              multidimensional=True, normalize_x=True)
    save_plot(plot_location, 'losses_normalized', listifydict(PLOTS["LOSSES"]),
              'Batch index', 'Loss ratio',
              multidimensional=True, normalize_x=True, normalize_y=True)

    # Plots of max/only values
    logline('')
    logline('Plotting max/only values')
    save_plot(plot_location, 'time_since_last_access', listifydict(PLOTS["TIME_SINCE_LAST_ACCESS"], get_max=True),
              'User index', 'Max time since last access (in seconds)')
    save_plot(plot_location, 'unique_domains', listifydict(PLOTS["UNIQUE_DOMAINS"], get_max=True),
              'User index', 'Max unique domains')
    save_plot(plot_location, 'unique_dest_users', listifydict(PLOTS["UNIQUE_DEST_USERS"], get_max=True),
              'User index', 'Max dest users')
    save_plot(plot_location, 'percentage_failed_logins', listifydict(PLOTS["PERCENTAGE_FAILED_LOGINS"], get_max=True),
              'User index', 'Percentage failed logins')
    save_plot(plot_location, 'iqrs', listifydict(PLOTS["IQRS"]),
              'Batch index', 'Interquartile range')
    save_plot(plot_location, 'iqr_maxes', listifydict(PLOTS["IQR_MAXES"]),
              'Batch index', 'Max interquartile range')

    # Non-plots, just logs
    logline('')
    logline('Saving data-only values')
    save_plot(plot_location, 'timings_training', PLOTS["USER_TIMINGS_TRAINING"],
              '', '', is_plot=False)
    save_plot(plot_location, 'timings_test', PLOTS["USER_TIMINGS_TEST"],
              '', '', is_plot=False)


def is_closer(target: int, a: int, b: int) -> bool:
    return abs(target - a) <= abs(target - b)


T = TypeVar('T')


def get_user_list(orig_list: List[T], start: int, end: int) -> List[T]:
    if not USES_DIFFERENT_INDEXES:
        return orig_list

    total_samples = 0
    start_indexes = list()
    for i in range(len(orig_list)):
        user = orig_list[i]
        length = len(user["datasets"]["training"])
        total_samples += length
        start_indexes.append({
            "length": total_samples - length,
            "index": i
        })

    logline('There are a total of', total_samples, 'rows')

    # Get start and end sample lengths
    start_sample_index = round(float(total_samples) * (start / 100))
    end_sample_index = round(float(total_samples) * (end / 100))

    final_start_index = -1
    final_end_index = -1

    # Locate the closest start/end of a sample
    for i in range(len(start_indexes)):
        data = start_indexes[i]
        if final_start_index == -1 and data["length"] > start_sample_index:
            # Check if the last one may have been closer
            if i != 0 and is_closer(start_sample_index, start_indexes[i - 1]["length"], data["length"]):
                final_start_index = start_indexes[i - 1]["index"]
            else:
                final_start_index = data["index"]
        if final_end_index == -1 and data["length"] > end_sample_index:
            # Check if the last one may have been closer
            if i != 0 and is_closer(end_sample_index, start_indexes[i - 1]["length"], data["length"]):
                final_end_index = start_indexes[i - 1]["index"]
            else:
                final_end_index = data["index"]
        if final_end_index != -1 and final_start_index != -1:
            break

    if final_start_index == -1:
        final_start_index = 0
    if final_end_index == -1:
        final_end_index = len(orig_list) - 1

    return orig_list[final_start_index:final_end_index]


def open_users_list():
    with open(io.get('input_file'), 'rb') as in_file:
        full_list = pickle.load(in_file)

    total_users = len(full_list)
    logline('Found a total of', total_users, 'users')
    logline("Dividing list")
    divided = get_user_list(full_list, io.get('start'), io.get('end'))
    logline("There are", total_users, "users, and this process is doing", len(divided), "of them")
    return divided


def do_detection(model: RNNModel, users_list: List[Dict[str, Union[str, Dict[str, List[List[float]]]]]]
                 ) -> Dict[str, List[Dict[str, int]]]:
    logline('Calculating total dataset size')
    total_samples = 0
    for user in users_list:
        total_samples += len(user["datasets"]["training"])

    timer = Timer(total_samples)

    logline("Starting anomaly detection")

    all_anomalies = dict()
    tested_users = 0
    for user in users_list:

        logline("")
        percentage = round((tested_users * 100) / len(users_list))
        logline("Checking user ", tested_users, "/", len(users_list),
                " (", percentage, "%)", spaces_between=False)
        logline("ETA is " + timer.get_eta())

        current_user = Dataset(user)

        try:
            anomalies = find_anomalies(model, current_user)
            if len(anomalies) > 0:
                all_anomalies[current_user.user_name] = anomalies
            tested_users += 1
            timer.add_to_current(len(current_user.datasets.training))
        except KeyboardInterrupt:
            # Skip rest of users, report early
            logline("\n\nSkipping rest of the users")
            break

    debug('Runtime is', timer.report_total_time())
    return all_anomalies


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main():
    """The main function"""
    if not io.run:
        return

    plot_location = io.get('plot_location')

    users_list = open_users_list()

    try:
        logline("Setting up generic model...")
        model = RNNModel()
    except tf.errors.InternalError:
        error("No GPU is currently available for you, aborting")
        raise

    all_anomalies = do_detection(model, users_list)

    logline("Done checking users, outputting results now")

    if plot_location is not None:
        try:
            if not os.path.exists(plot_location):
                os.makedirs(plot_location)
        except OSError as exception:
            error("Error", exception)

        logline("Plotting various things in", plot_location)
        save_plot_data(plot_location)

    output_file = io.get('output_file')
    if output_file == 'stdout':
        logline("Outputting results to stdout\n\n\n")
        logline(json.dumps(all_anomalies, cls=NumpyEncoder))
    else:
        try:
            out_file_parts = output_file.split('/')
            dir_path = '/'.join(out_file_parts[:-1])
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        except OSError as exception:
            error("Error", exception.errno)

        if USES_DIFFERENT_INDEXES:
            output_file = output_file[0:-5] + '.part.' + str(io.get('start')) + '.' + str(io.get('end')) + '.json'
        logline("Outputting results to", output_file)
        with open(output_file, 'w') as out_file:
            debug('All_anomalies is', all_anomalies)
            out_file.write(json.dumps(all_anomalies, cls=NumpyEncoder))

    logline("Done, closing files and stuff")


if __name__ == "__main__":
    exit_code = 0
    start_time = time.time()
    try:
        main()
    except Exception as e:
        logline("An exception has occurred", "\n",
                traceback.format_exc())
        exit_code = 1
    else:
        logline('Ran successfully')
    finally:
        logline('Total runtime is', Timer.stringify_time(Timer.format_time(time.time() - start_time)))
        log_to_folder_done()
        try:
            sys.exit(exit_code)
        except AttributeError:
            logline("Tensorflow threw some error while closing, just ignore it")
