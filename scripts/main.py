"""The main script file"""
import os
import _pickle as pickle
import sys
import math
import json
import tensorflow as tf
from typing import List, Dict, Tuple, Union, TypeVar
import matplotlib.pyplot as plt
import numpy as np
import features
from imports.timer import Timer
from imports.log import logline
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
    'm': IOInput(1.2, float, arg_name='magic_number',
                 descr='The magic number at which a test sample becomes an anomaly',
                 alias='magic_number'),
    'x': IOInput(True, bool, has_input=False, descr='Disable verbose output during running',
                 alias='running_verbose'),
    'p': IOInput(None, str, arg_name='plot_location', descr="The location to store the plot of the losses (not "
                                                            "specifying a location skips plotting)",
                 alias='plot_location'),
    'M': IOInput(False, bool, has_input=False, descr='Enable meganet mode',
                 alias='meganet')
})


EPOCHS = io.get('epochs')
GIVE_TEST_SET_PREVIOUS_KNOWLEDGE = io.get('give_prev_knowledge')
VERBOSE = io.get('verbose')
VERBOSE_RUNNING = io.get('running_verbose')

MAGIC_NUMBER = io.get('magic_number')


if io.run:
    from keras.layers.core import Dense
    from keras.layers.recurrent import LSTM
    from keras.models import Sequential

plt.switch_backend('agg')

np.random.seed(1234)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LSTM_SIZE = 2 ** 4
BATCH_SIZE = 32
SPEED_REPORTING_SIZE = 1000
ANOMALY_THRESHOLD = 1.0
CONTEXT_LENGTH = 10

LOSSES = list()
FEATURE_SET = List[List[float]]


def force_str(val: str) -> str:
    return val


def force_feature(val: List[float]) -> List[float]:
    return val


def force_feature_set(val: FEATURE_SET) -> FEATURE_SET:
    return val


class DataDistribution:
    def __init__(self, data: Dict[str, FEATURE_SET]):
        self.training = force_feature_set(data["training"])
        self.test = force_feature_set(data["test"])


class Dataset:
    def __init__(self, data: Dict[str, Union[str, List[float], Dict[str, FEATURE_SET]]]):
        self.user_name = force_str(data["user_name"])
        self.datasets = DataDistribution(data["datasets"])


class FeatureDescriptor:
    def __init__(self, name: str, type_: str, weight: float):
        self.name = name
        self.type = type_
        self.weight = weight


FEATURE_SIZE = features.size()
LAYERS = [FEATURE_SIZE, 4, 4, FEATURE_SIZE]


def create_anomaly(start: int, end: int) -> Dict[str, int]:
    return {
        "start": start,
        "end": end
    }


class RNNModel:
    """An RNN"""

    def __init__(self):
        model = Sequential()
        model.add(LSTM(LAYERS[1], input_shape=(FEATURE_SIZE, 1), batch_size=BATCH_SIZE,
                       return_sequences=True, stateful=True))
        model.add(LSTM(LAYERS[2], return_sequences=False, stateful=True))
        model.add(Dense(LAYERS[3]))
        model.compile(loss='mean_squared_error', optimizer='adam')

        self.model = model

        self._starting_weights = list()
        for i in range(len(model.layers)):
            self._starting_weights.append(model.layers[i].get_weights())

    @staticmethod
    def prepare_data(training_data: FEATURE_SET, test_data=None):
        """Prepares given datasets for insertion into the model"""

        if len(training_data) == 1:
            print(training_data, test_data)
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
                print("Epoch", i, '/', epochs)
            verbosity = 0
            if VERBOSE_RUNNING:
                verbosity = 1
            self.model.fit(train_x, train_y, batch_size=BATCH_SIZE,
                           epochs=1, verbose=verbosity, shuffle=False)
            if not GIVE_TEST_SET_PREVIOUS_KNOWLEDGE or i != epochs - 1:
                self.model.reset_states()

    def test(self, test_x, test_y) -> List[float]:
        """Predicts the result for given test data"""
        losses = list()
        for i in range(round(len(test_x) / BATCH_SIZE)):
            losses.append(self.model.evaluate(test_x[i * BATCH_SIZE:(i + 1) * BATCH_SIZE],
                                              test_y[i * BATCH_SIZE:(i + 1) * BATCH_SIZE],
                                              batch_size=BATCH_SIZE, verbose=0))
        return losses


def abs_ratio(a: float, b: float) -> float:
    if b == 0:
        return 1000
    ratio = a / b
    if ratio < 1:
        return 1 / ratio
    return ratio


def mean(data: List[float]) -> float:
    copy = data[:]
    copy.sort()

    if len(copy) % 2 == 0:
        half = round(math.floor(len(copy) / 2))
        return (copy[half - 1] + copy[half]) / 2
    else:
        return copy[round(len(copy) / 2)]


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

    def get_losses(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        return self.model.test(x, y)

    def find_anomalies(self) -> List[Dict[str, int]]:
        # Train the network first
        train_x, train_y, test_x, test_y = RNNModel.prepare_data(self.dataset.training, test_data=self.dataset.test)
        self.model.fit(train_x, train_y, epochs=self.config["epochs"])

        print("\nChecking losses on test set...")
        test_losses = self.get_losses(test_x, test_y)
        print("Done checking losses on test set\n")

        anomalies = list()

        mean_loss = mean(test_losses)

        global LOSSES
        LOSSES.append(test_losses)

        for i in range(len(test_losses)):
            if test_losses[i] >= MAGIC_NUMBER * mean_loss:
                anomaly = create_anomaly(len(train_x) + i * BATCH_SIZE, len(train_x) + (i + 1) * BATCH_SIZE)
                anomalies.append(anomaly)

        return anomalies


def find_anomalies(model: RNNModel, data: Dataset) -> List[Dict[str, int]]:
    """Finds anomalies in given data"""
    network = UserNetwork(model, data, epochs=EPOCHS)
    anomalies = network.find_anomalies()
    return anomalies


def plot_losses(plot_location: str):
    """Plots all the losses and saves them"""

    # Get biggest sample size to spread out against
    biggest_sample_size = 0
    for i in range(len(LOSSES)):
        if len(LOSSES[i]) > biggest_sample_size:
            biggest_sample_size = len(LOSSES[i])

    print("Gathering data points and plotting")
    for i in range(len(LOSSES)):
        sample_size = len(LOSSES[i])
        scalar = biggest_sample_size / sample_size

        x_values = list()
        y_values = list()

        for j in range(len(LOSSES[i])):
            x_values.append(scalar * j)
            y_values.append(LOSSES[i][j])

        plt.plot(x_values, y_values, markersize=1)

    # plt.yscale('log')
    plt.ylabel('Loss ratio')
    plt.xlabel('Batch index')
    plt.savefig(plot_location)
    print("Saved plot to", plot_location)


def is_closer(target: int, a: int, b: int) -> bool:
    return abs(target - a) <= abs(target - b)


T = TypeVar('T')


def get_user_list(orig_list: List[T], start: int, end: int) -> Tuple[List[T], bool]:
    if start == 0 and end == 100:
        return orig_list, False

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

    return orig_list[final_start_index:final_end_index], True


def open_users_list():
    with open(io.get('input_file'), 'rb') as in_file:
        full_list = pickle.load(in_file)

    if not io.get('meganet'):
        total_users = len(full_list['training'])
        logline("Dividing list...")
        divided = get_user_list(full_list, io.get('start'), io.get('end'))
        logline("There are", total_users, "users, and this process is doing", len(divided[0]), "of them")
        return divided
    else:
        total_rows = len(full_list['training'])
        logline('There are', total_rows, 'events')
        return full_list, False


def do_non_megalist_detection(model: RNNModel, users_list: List[Dict[str, Union[str, Dict[str, List[List[float]]]]]]
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

        if tested_users > 0:
            print("\nChecking user", tested_users, "/", len(users_list), " - ETA is " + timer.get_eta())

        current_user = Dataset(user)

        try:
            anomalies = find_anomalies(model, current_user)
            if len(anomalies) > 0:
                all_anomalies[current_user.user_name] = anomalies
            tested_users += 1
            timer.add_to_current(len(current_user.datasets.training))
        except KeyboardInterrupt:
            # Skip rest of users, report early
            print("\n\nSkipping rest of the users")
            break

    return all_anomalies


def train_on_batch(model: RNNModel, batch: List[List[float]]):
    train_x, train_y = RNNModel.prepare_data(batch)
    model.fit(train_x, train_y, epochs=io.get('epochs'))


def find_meganet_anomalies(model: RNNModel, batch: List[List[float]]) -> List[Dict[str, int]]:
    model.reset(reset_weights=False)

    test_x, test_y = RNNModel.prepare_data(batch)
    test_losses = model.test(test_x, test_y)

    anomalies = list()

    mean_loss = mean(test_losses)

    global LOSSES
    LOSSES.append(test_losses)

    for i in range(len(test_losses)):
        if test_losses[i] >= MAGIC_NUMBER * mean_loss:
            anomaly = create_anomaly(i * BATCH_SIZE, (i + 1) * BATCH_SIZE)
            anomalies.append(anomaly)

    return anomalies


def do_megalist_detection(model: RNNModel, dataset: Dict[str, Union[List[Union[List[float], int]],
                                                                    List[Dict[str, Union[str, FEATURE_SET]]]]]
                          ) -> Dict[str, List[Dict[str, int]]]:
    logline("Starting anomaly detection")

    all_anomalies = dict()
    tested_users = 0

    training_arr = np.array(dataset['training'])
    training_set_length = len(training_arr)
    training_timer = Timer(training_set_length)

    #Reduce by one for logging clarity later
    training_set_length -= 1

    for i in range(5):
        user_list = training_arr[i]
        if tested_users > 0:
            logline("\nTraining on user", tested_users, "/", training_set_length, "- ETA for training is " +
                    training_timer.get_eta())
        try:
            train_on_batch(model, user_list)
            tested_users += 1
            training_timer.add_to_current(1)
        except KeyboardInterrupt:
            # Skip rest of users, report early
            logline("\n\nSkipping rest of the users")
            break

    tested_users = 0
    test_set_length = len(dataset['test'])

    # Reduce by one for logging clarity later
    test_set_length -= 1

    testing_timer = Timer(test_set_length)
    for user in dataset['test']:
        if tested_users > 0:
            logline("\nTesting user", tested_users, "/", test_set_length, " - ETA for testing is " +
                    testing_timer.get_eta())

        try:
            anomalies = find_meganet_anomalies(model, user['dataset'])
            if len(anomalies) > 0:
                all_anomalies[user['user_name']] = anomalies
            tested_users += 1
            test_set_length.add_to_current(1)
        except KeyboardInterrupt:
            # Skip rest of users, report early
            logline("\n\nSkipping rest of the users")
            break

    return all_anomalies


def main():
    """The main function"""
    if not io.run:
        return

    plot_location = io.get('plot_location')

    users_list, uses_different_indexes = open_users_list()

    try:
        logline("Setting up generic model")
        model = RNNModel()
    except tf.errors.InternalError:
        logline("No GPU is currently available for you, aborting")
        raise

    is_meganet = io.get('meganet')
    if is_meganet:
        all_anomalies = do_megalist_detection(model, users_list)
    else:
        all_anomalies = do_non_megalist_detection(model, users_list)

    logline("Done checking users, outputting results now")

    if plot_location is not None:
        logline("Plotting losses")
        plot_losses(plot_location)

    output_file = io.get('output_file')
    if output_file == 'stdout':
        logline("Outputting results to stdout\n\n\n")
        logline(json.dumps(all_anomalies))
    else:
        if uses_different_indexes:
            output_file = output_file[0:-5] + '.part.' + str(io.get('start')) + '.' + str(io.get('end')) + '.json'
        logline("Outputting results to", output_file)
        with open(output_file, 'w') as out_file:
            out_file.write(json.dumps(all_anomalies))

    logline("Done, closing files and stuff")
    try:
        sys.exit()
    except AttributeError:
        logline("Tensorflow threw some error while closing, just ignore it")


if __name__ == "__main__":
    main()
