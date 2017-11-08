"""The main script file"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PIC_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import _pickle as pickle
import warnings
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
from sklearn.metrics import mean_squared_error
import math

warnings.simplefilter(action='ignore', category=FutureWarning)

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
                 descr='The folder to output the logs',
                 alias='log_folder'),
    'e': IOInput(None, str, arg_name='experiment',
                 descr='The experiment to run',
                 alias='experiment')
})


# Constants
BATCH_SIZE = 32
SPEED_REPORTING_SIZE = 1000
MAX_HIGHEST_OFFENDERS = 10
ANOMALY_HISTORY = 31
DROPOUT = 0.5
RECURRENT_DROPOUT = 0.2
LEARNING_RATE = 0.001

# Global variables/functions
EPOCHS = 25
GIVE_TEST_SET_PREVIOUS_KNOWLEDGE = io.get('give_prev_knowledge')
VERBOSE = io.get('verbose')
VERBOSE_RUNNING = io.get('running_verbose')
USES_DIFFERENT_INDEXES = io.get('start') != 0 or io.get('end') != 100
logline, debug, error, log_to_folder_done = logline_to_folder(folder_loc=io.get('log_folder'),
                                                              file_name='main_logs.log',
                                                              start=io.get('start'), end=io.get('end'))
T = TypeVar('T')

if io.run:
    import tensorflow as tf
    import keras
    from keras import optimizers
    from keras.layers.core import Dense
    from keras.layers.recurrent import LSTM, SimpleRNN, GRU
    from keras.models import Sequential

plt.switch_backend('agg')

np.random.seed(1234)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PLOTS = {
    "LOSSES": dict(),
    "TIME_SINCE_LAST_ACCESS": dict(),
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
PERCENTAGE_FAILED_LOGINS_INDEX = 5


def bigger_batch_size():
    global BATCH_SIZE
    BATCH_SIZE = 64


def smaller_batch_size():
    global BATCH_SIZE
    BATCH_SIZE = 16


def more_epochs():
    global EPOCHS
    EPOCHS = 40


def less_epochs():
    global EPOCHS
    EPOCHS = 15


def more_layers():
    global EXPERIMENTS_MAP
    EXPERIMENTS_MAP["more_layers"] = True


def less_layers():
    global EXPERIMENTS_MAP
    EXPERIMENTS_MAP["less_layers"] = True


def hidden_nodes_increase():
    global LAYERS
    LAYERS[1] = FEATURE_SIZE * 2
    LAYERS[2] = math.floor(FEATURE_SIZE * 1.5)


def regular_rnn():
    global EXPERIMENTS_MAP
    EXPERIMENTS_MAP["regular_rnn"] = True


def gru_rnn():
    global EXPERIMENTS_MAP
    EXPERIMENTS_MAP["gru_rnn"] = True


def lower_dropout():
    global EXPERIMENTS_MAP
    EXPERIMENTS_MAP["lower_dropout"] = True


def higher_learning_rate():
    global LEARNING_RATE
    LEARNING_RATE = LEARNING_RATE * 2


def lower_learning_rate():
    global LEARNING_RATE
    LEARNING_RATE = LEARNING_RATE / 2


def different_optimizer():
    global EXPERIMENTS_MAP
    EXPERIMENTS_MAP["different_optimizer"] = True


def clear_state_on_test():
    global GIVE_TEST_SET_PREVIOUS_KNOWLEDGE
    GIVE_TEST_SET_PREVIOUS_KNOWLEDGE = False


EXPERIMENTS = {
    "bigger_batch_size": bigger_batch_size,
    "smaller_batch_size": smaller_batch_size,
    "more_epochs": more_epochs,
    "less_epochs": less_epochs,
    "more_layers": more_layers,
    "less_layers": less_layers,
    "hidden_nodes_increase": hidden_nodes_increase,
    "regular_rnn": regular_rnn,
    "gru_rnn": gru_rnn,
    "lower_dropout": lower_dropout,
    "higher_learning_rate": higher_learning_rate,
    "lower_learning_rate": lower_learning_rate,
    "different_optimizer": different_optimizer,
    "clear_state_on_test": clear_state_on_test
}

EXPERIMENTS_MAP = { }


def force_str(val: str) -> str:
    return val


def force_feature(val: List[float]) -> List[float]:
    return val


def force_feature_set(val: FEATURE_SET) -> FEATURE_SET:
    return val


def force_list_of_int(val: List[int]) -> List[int]:
    return val

def val_at_index(val: List[List[float]], index: int) -> List[float]:
    return list(map(lambda x: x[index], val))


class DataDistribution:
    def __init__(self, data: Dict[str, Union[FEATURE_SET, List[float]]], user_name: str):
        self.training = force_feature_set(data["training"])
        self.test = force_feature_set(data["test"])
        self.scale = force_feature(data['scales'])

        PLOTS["TIME_SINCE_LAST_ACCESS"][user_name] = {
            "max": max(max(val_at_index(self.training, TIME_SINCE_LAST_ACCESS_INDEX)),
                       max(val_at_index(self.test, TIME_SINCE_LAST_ACCESS_INDEX))) *
                   self.scale[TIME_SINCE_LAST_ACCESS_INDEX],
            "all": list(map(lambda x: x * self.scale[TIME_SINCE_LAST_ACCESS_INDEX],
                            list(np.array(self.test)[:, TIME_SINCE_LAST_ACCESS_INDEX])))
        }
        PLOTS["PERCENTAGE_FAILED_LOGINS"][user_name] = {
            "all": list(map(lambda x: x * self.scale[PERCENTAGE_FAILED_LOGINS_INDEX],
                        list(np.array(self.test)[:, PERCENTAGE_FAILED_LOGINS_INDEX])))
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
LAYERS = [FEATURE_SIZE, FEATURE_SIZE, FEATURE_SIZE, FEATURE_SIZE]


def create_anomaly(start: int = None, end: int = None, train_len: int = None, dataset: FEATURE_SET = None,
                   predicted: List[float] = None, actual: List[float] = None, loss: float = None) -> Dict[str, Union[int, List[float]]]:
    if None in [start, end, train_len, dataset, predicted, actual, loss]:
        error('One of create_anomaly\'s arguments was not supplied')
        sys.exit(2)

    final_row = dataset[end - 1]
    return {
        "start": start + train_len,
        "end": end + train_len,
        "final_row_features": final_row,
        "predicted": predicted,
        "actual": actual,
        "loss": loss
    }


def rnn_model(batch_size=BATCH_SIZE):
    model = Sequential()
    if "regular_rnn" in EXPERIMENTS_MAP:
        layer = SimpleRNN
    elif "gru_rnn" in EXPERIMENTS_MAP:
        layer = GRU
    else:
        layer = LSTM

    model.add(layer(LAYERS[1], input_shape=(LAYERS[0], 1), batch_size=batch_size,
                   return_sequences=True, stateful=True, dropout=DROPOUT,
                    recurrent_dropout=RECURRENT_DROPOUT))
    if "more_layers" in EXPERIMENTS_MAP:
        model.add(layer(LAYERS[2], batch_size=batch_size,
                        return_sequences=True, stateful=True, dropout=DROPOUT,
                        recurrent_dropout=RECURRENT_DROPOUT))
    if not "less_layers" in EXPERIMENTS_MAP:
        model.add(layer(LAYERS[2], batch_size=batch_size,
                        return_sequences=False, stateful=True, dropout=DROPOUT,
                        recurrent_dropout=RECURRENT_DROPOUT))
    model.add(Dense(LAYERS[3], activation="relu"))

    if "different_optimizer" in EXPERIMENTS_MAP:
        optimizer = optimizers.adam(lr=LEARNING_RATE)
    else:
        optimizer = optimizers.SGD()

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


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

        test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

        return train_x, train_y, test_x, test_y
    return train_x, train_y


class TrainingSession:
    """A training session featuring a RNN with batch_size BATCH_SIZE"""

    def __init__(self, model, batch_size=BATCH_SIZE):
        self.model = model
        self.batch_size = batch_size

        self._starting_weights = list()
        for i in range(len(model.layers)):
            self._starting_weights.append(model.layers[i].get_weights())

    def reset(self, reset_weights=True):
        if reset_weights:
            for i in range(len(self.model.layers)):
                self.model.layers[i].set_weights(self._starting_weights[i])
        self.model.reset_states()

    def fit(self, train_x, train_y, epochs: int = 10):
        """Fits the model to given training data"""

        self.model.reset_states()

        verbosity = 1 if VERBOSE_RUNNING else 0
        for i in range(epochs):
            logline('Epoch ', i + 1, '/', epochs, spaces_between=False)
            self.model.fit(train_x, train_y, batch_size=self.batch_size, epochs=1,
                           verbose=verbosity, shuffle=False)
            self.model.reset_states()


class TestLoss:
    def __init__(self, loss: float, prediction: List[float], actual: List[float]):
        self.loss = loss
        self.prediction = prediction
        self.actual = actual


class TestSession:
    """A testing session for an RNN with batch_size 1"""

    def __init__(self, model):
        self.model = model
        self.batch_size = 1

        self._starting_weights = list()
        for i in range(len(model.layers)):
            self._starting_weights.append(model.layers[i].get_weights())

    def reset(self, reset_weights=True):
        if reset_weights:
            for i in range(len(self.model.layers)):
                self.model.layers[i].set_weights(self._starting_weights[i])
        self.model.reset_states()

    def test(self, test_x, test_y) -> List[TestLoss]:
        """Predicts the result for given test data"""
        if not GIVE_TEST_SET_PREVIOUS_KNOWLEDGE:
            self.model.reset_states()

        losses = list()
        predictions = self.model.predict(test_x, batch_size=self.batch_size, verbose=0)
        for i in range(len(predictions)):
            prediction = predictions[i]
            actual = test_y[i]

            loss = mean_squared_error(actual, prediction)
            losses.append(TestLoss(loss, prediction, actual))

        return losses


class Session:
    def __init__(self, train: TrainingSession, test: TestSession):
        self.train = train
        self.test = test


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
    return (value - q3) / inter_quartile_range


class UserNetwork:
    """The class describing a single model and all its corresponding data"""

    def __init__(self, training_model: TrainingSession, test_model: TestSession, data: Dataset, epochs: int = 10):
        """Creates a new set of networks"""

        self.user_name = data.user_name
        self.dataset = data.datasets
        self.config = {
            "epochs": epochs
        }

        self.train_model = training_model
        self.test_model = test_model

        self.train_model.reset()
        self.test_model.reset()

    def sync_models(self):
        # Sync weights
        self.test_model.model.set_weights(self.train_model.model.get_weights())

    @staticmethod
    def get_training_set_iqr(train_x: List[List[float]], train_y: List[List[float]]):
        mses = list()
        for i in range(len(train_x)):
            mses.append(mean_squared_error(train_x[i], train_y[i]))

        return iqr(mses)

    def find_anomalies(self) -> List[Dict[str, Union[int, List[float]]]]:
        # Train the network first
        train_x, train_y, test_x, test_y = prepare_data(self.dataset.training, test_data=self.dataset.test)
        self.train_model.fit(train_x, train_y, epochs=self.config["epochs"])

        logline("\n")
        logline('Syncing models')
        self.sync_models()
        logline("Checking losses on test set...")
        test_losses = self.test_model.test(test_x, test_y)
        logline("Done checking losses on test set\n")

        anomalies = list()

        # Interquartile range
        inter_quartile_range, q3 = self.get_training_set_iqr(train_x, train_y)

        PLOTS["LOSSES"][self.user_name] = list(map(lambda x: x.loss, test_losses))
        PLOTS["IQRS"][self.user_name] = inter_quartile_range
        PLOTS["IQR_MAXES"][self.user_name] = q3 + (1.5 * inter_quartile_range)

        for i in range(len(test_losses)):
            test_loss = test_losses[i]
            iqr_distance = get_iqr_distance(inter_quartile_range, q3, test_loss.loss)
            if iqr_distance > 1.5:
                anomaly = create_anomaly(start=i - ANOMALY_HISTORY,
                                         end=i + 1,
                                         dataset=test_y,
                                         train_len=len(train_x),
                                         actual=test_loss.actual,
                                         predicted=test_loss.prediction,
                                         loss=test_loss.loss)
                anomalies.append(anomaly)

                PLOTS["DEVIATIONS"].append({
                    "key": self.user_name,
                    "val": iqr_distance
                })
            PLOTS["ALL_DEVIATIONS"].append(iqr_distance)

        return anomalies


def find_anomalies(session: Session, data: Dataset) -> List[Dict[str, Union[int, List[float]]]]:
    """Finds anomalies in given data"""
    network = UserNetwork(session.train, session.test, data, epochs=EPOCHS)
    anomalies = network.find_anomalies()
    return anomalies


def save_plot(plot_location: str, name: str,
              data: Any,
              x_label: str, y_label: str,
              is_log: bool = False, normalize_x: bool = False, normalize_y: bool = False,
              multidimensional: bool = False, is_dict=False, is_box_plot=False, is_highest_offenders=False,
              is_plot: bool = True, is_sorted: bool = False):
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
        "is_sorted": is_sorted,
        "is_box_plot": is_box_plot,
        "is_highest_offenders": is_highest_offenders,
        "max_highest_offenders": MAX_HIGHEST_OFFENDERS,
        "data": data
    }

    if USES_DIFFERENT_INDEXES:
        plot_location = plot_location + name + '.part.' + str(io.get('start')) + '.' + str(io.get('end')) + '.json'
    else:
        plot_location = plot_location + name + '.json'
    logline("Outputting plot data to", plot_location)
    with open(plot_location, 'w') as out_file:
        out_file.write(json.dumps(plot_data))


def listifydict(data_dict: Dict[str, T], get_max=False, get_all=False, get_last=False) -> List[T]:
    data_list = list()
    for key, val in data_dict.items():
        if get_max:
            data_list.append(val["max"])
        elif get_all:
            data_list.append(val["all"])
        elif get_last:
            data_list.append(val["all"][-1])
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


def data_points_for_users(highest_offenders: List[Dict[str, Union[str, float, Dict[str, Union[str, float]]]]],
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


def filter_deviations(deviations):
    by_user = dict()
    for i in range(len(deviations)):
        data = deviations[i]
        user = data["key"]
        val = data["val"]
        if user in by_user:
            by_user[user].append(val)
        else:
            by_user[user] = [val]
    
    filtered_deviations = list()
    for user, datapoints in by_user.items():
        last_five = datapoints[-5:]
        for i in range(len(last_five)):
            filtered_deviations.append({
                "key": user,
                "val": last_five[i]
            })
    return filtered_deviations


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
              'Action index', 'Deviation from IQR (y)',
              is_log=True, is_sorted=True)
    save_plot(plot_location, 'filtered_deviations', list(map(lambda x: x["val"], filter_deviations(PLOTS["DEVIATIONS"]))),
              'Action index', 'Deviation from IQR (y)',
              is_log=True, is_sorted=True)
    with open(plot_location + 'deviations_original.json', 'w') as deviations_out:
        json.dump(deviations_out)
    save_plot(plot_location, 'highest_offender_deviations', highest_offenders_with_mean_dict,
              'User Name', 'Relative deviation (from mean)',
              is_dict=True, multidimensional=True, is_highest_offenders=True, is_log=True)
    save_plot(plot_location, 'highest_offender_time_since_last_access',
              data_points_for_users(highest_offenders, PLOTS["TIME_SINCE_LAST_ACCESS"], key="all"),
              'User Name', 'Time since last access (in seconds)',
              is_dict=True, is_box_plot=True, is_highest_offenders=True)
    save_plot(plot_location, 'highest_offender_failed_logins',
              data_points_for_users(highest_offenders, PLOTS["PERCENTAGE_FAILED_LOGINS"], key="all"),
              'User Name', 'Percentage failed logins',
              is_dict=True, is_box_plot=True, is_highest_offenders=True)

    # Plots of the losses of all batches
    logline('')
    logline('Plotting losses')
    save_plot(plot_location, 'all deviations', PLOTS["ALL_DEVIATIONS"],
              'Batch index', 'IQR Scale', is_log=True, is_sorted=True)
    save_plot(plot_location, 'losses', listifydict(PLOTS["LOSSES"]),
              'Batch index', 'Loss ratio',
              multidimensional=True, normalize_x=True)

    # Plots of max/only values
    logline('')
    logline('Plotting max/only values')
    save_plot(plot_location, 'time_since_last_access', listifydict(PLOTS["TIME_SINCE_LAST_ACCESS"], get_max=True),
              'User index', 'Max time since last access (in seconds)')
    save_plot(plot_location, 'time_since_last_access_log', listifydict(PLOTS["TIME_SINCE_LAST_ACCESS"], get_max=True),
              'User index', 'Max time since last access (in seconds)', is_log=True)
    save_plot(plot_location, 'percentage_failed_logins', listifydict(PLOTS["PERCENTAGE_FAILED_LOGINS"], get_last=True),
              'User index', 'Total percentage failed logins')
    save_plot(plot_location, 'iqrs', listifydict(PLOTS["IQRS"]),
              'Batch index', 'Interquartile range',
              is_sorted=True)
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


def do_detection(session: Session, users_list: List[Dict[str, Union[str, Dict[str, List[List[float]]]]]]
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
            anomalies = find_anomalies(session, current_user)
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


def select_experiment():
    experiment = io.get('experiment')
    if experiment:
        EXPERIMENTS[experiment]()


def main():
    """The main function"""
    if not io.run:
        return

    select_experiment()

    plot_location = io.get('plot_location')

    logline('opening users list')
    users_list = open_users_list()

    try:
        logline("Setting up generic models...")
        train_model = TrainingSession(rnn_model())
        test_model = TestSession(rnn_model(batch_size=1))

        debug('Training model summary is:')
        train_model.model.summary()
        debug('')
        debug('')

        debug('Test model summary is:')
        test_model.model.summary()
        debug('')
        debug('')


        session = Session(train_model, test_model)
    except tf.errors.InternalError:
        error("No GPU is currently available for you, aborting")
        raise

    all_anomalies = do_detection(session, users_list)

    logline("Done checking users, outputting results now")

    if plot_location is not None:
        try:
            if not os.path.exists(plot_location):
                os.makedirs(plot_location)
        except OSError as exception:
            error("Error", exception)

        logline("Plotting various things in", plot_location)
        try:
            save_plot_data(plot_location)
        except Exception as e:
            error('Something went wrong creating plots, dumping to file')
            try:
                with open(plot_location + 'plot_data.json', 'w') as out_file:
                    json.dump(PLOTS, out_file)
            except Exception as e:
                error('Something went even more wrong when outputting to file, dumping to stdout')
                try:
                    data = json.dumps(PLOTS)
                    debug(data)
                except Exception as e:
                    error('Something went wrong JSON stringifying the data, dumping obj')
                    debug(PLOTS)

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
            try:
                out_file.write(json.dumps(all_anomalies, cls=NumpyEncoder))
            except Exception:
                error('Could not write to output for some reason, printing instead')
                debug('All_anomalies is', all_anomalies)

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
