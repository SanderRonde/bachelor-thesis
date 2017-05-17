"""The main script file"""
import getopt
import os
import _pickle as pickle
import sys
import time
import math
import json
import tensorflow as tf
from typing import List, Dict, Tuple, Union, TypeVar
import matplotlib.pyplot as plt
import numpy as np


EPOCHS = 25
GIVE_TEST_SET_PREVIOUS_KNOWLEDGE = True
VERBOSE = True
VERBOSE_RUNNING = True


def get_io_settings(argv: sys.argv) -> Tuple[str, str, str, int, int]:
    """This gets the input and output files from the command line arguments"""
    input_file = '/data/s1495674/features.p'
    output_file = '/data/s1495674/anomalies.encoded.json'
    plot_location = None
    start_distr = 0
    end_distr = 100

    try:
        opts, args = getopt.getopt(argv, 'i:o:p:e:d:s:m:khvx')
    except getopt.GetoptError:
        print("Command line arguments invalid")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-i':
            input_file = arg
        elif opt == '-o':
            output_file = arg
        elif opt == '-p':
            plot_location = arg
        elif opt == '-v':
            global VERBOSE
            VERBOSE = True
        elif opt == '-k':
            global GIVE_TEST_SET_PREVIOUS_KNOWLEDGE
            GIVE_TEST_SET_PREVIOUS_KNOWLEDGE = not GIVE_TEST_SET_PREVIOUS_KNOWLEDGE
        elif opt == '-e':
            global EPOCHS
            EPOCHS = int(arg)
        elif opt == '-d':
            end_distr = int(arg)
        elif opt == '-m':
            global MAGIC_NUMBER
            MAGIC_NUMBER = float(arg)
        elif opt == '-s':
            start_distr = int(arg)
        elif opt == '-x':
            global VERBOSE_RUNNING
            VERBOSE_RUNNING = False
        elif opt == '-h':
            print("Options:")
            print(' -m <magic number>   The magic number at which a test sample becomes an anomaly')
            print(" -i <input file>     The source file for the users (in pickle format)")
            print(" -o <output file>    The file to output the anomalies to, specifying 'stdout' prints them to stdout")
            print(" -p <output file>    The location to store the plot of the losses (not specifying a location skips"
                  " plotting)")
            print(" -v                  Enable verbose output mode")
            print(" -k                  Specifying this disables keeping the training set in the state before trying "
                  "the test set")
            print(" -e <epochs>         The amount of epochs to use (default is " + str(EPOCHS) + ")")
            print(" -s <percentage>     The index at which to start processing")
            print(" -d <percentage>     The index at which to stop processing")
            print(' -x                  Disable verbose output during running')
            sys.exit()
        else:
            print("Unrecognized argument passed, refer to -h for help")
            sys.exit(2)

    return input_file, output_file, plot_location, start_distr, end_distr

IO = get_io_settings(sys.argv[1:])


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
MAGIC_NUMBER = 2.5

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


FEATURE_SIZE = 34
LAYERS = [FEATURE_SIZE, 4, 4, FEATURE_SIZE]


class Timer:
    """A timer to determine how long the entire operation might take"""

    def __init__(self, maximum: int):
        self._max = maximum
        self._current = 0
        self.start_time = time.time()

    def add_to_current(self, num: int):
        self._current += num

    def get_eta(self) -> str:
        passed_time = time.time() - self.start_time
        amount_done = self._current / self._max

        seconds = round(((1 / amount_done) * passed_time) - passed_time)
        if seconds <= 60:
            return str(seconds) + 's'

        mins = math.floor(seconds / 60)
        seconds = seconds % 60

        if mins <= 60:
            return str(mins) + 'm' + str(seconds) + 's'

        hours = math.floor(mins / 60)
        mins = mins % 60
        return str(hours) + 'h' + str(mins) + 'm' + str(seconds) + 's'


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
    def prepare_data(training_data: FEATURE_SET, test_data: FEATURE_SET):
        """Prepares given datasets for insertion into the model"""

        if len(training_data) == 1:
            print(training_data, test_data)
        assert len(training_data) > 1, "Training data is longer than 1, (is %d)" % len(training_data)
        assert len(test_data) > 1, "Test data is longer than 1, (is %d)" % len(test_data)

        train_x = np.array(training_data[:-1])
        train_y = np.array(training_data[1:])

        train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))

        test_x = np.array(test_data[:-1])
        test_y = np.array(test_data[1:])

        test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

        return train_x, train_y, test_x, test_y

    def reset(self):
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
    data.sort()

    if len(data) % 2 == 0:
        half = round(math.floor(len(data) / 2))
        return (data[half] + data[half + 1]) / 2
    else:
        return data[round(len(data) / 2)]


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
        train_x, train_y, test_x, test_y = RNNModel.prepare_data(self.dataset.training, self.dataset.test)
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
                anomaly = create_anomaly(len(train_x) +  i * BATCH_SIZE, len(train_x) + (i + 1) * BATCH_SIZE)
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


def main():
    """The main function"""

    global IO
    input_file, output_file, plot_location, start_distr, end_distr = IO

    with open(input_file, 'rb') as in_file:
        full_list = pickle.load(in_file)

    total_users = len(full_list)
    print("Dividing list...")
    users_list, uses_different_indexes = get_user_list(full_list, start_distr, end_distr)
    print("There are", total_users, "users, and this process is doing", len(users_list), "of them")

    try:
        print("Setting up generic model")
        model = RNNModel()
    except tf.errors.InternalError:
        print("No GPU is currently available for you, aborting")
        raise

    all_anomalies = dict()

    total_samples = 0
    print("Calculating total dataset size...")
    for user in users_list:
        total_samples += len(user["datasets"]["training"])

    timer = Timer(total_samples)

    print("Starting anomaly detection")

    tested_users = 0
    for user in users_list:

        if tested_users > 0:
            print("\nChecking user", tested_users, "/", len(users_list), "ETA is " + timer.get_eta())

        current_user = Dataset(user)

        try:
            anomalies = find_anomalies(model, current_user)
            if len(anomalies) > 0:
                all_anomalies[current_user] = anomalies
            tested_users += 1
            timer.add_to_current(len(current_user.datasets.training))
        except KeyboardInterrupt:
            # Skip rest of users, report early
            print("\n\nSkipping rest of the users")
            break

    print("Done checking users, outputting results now")

    if plot_location is not None:
        print("Plotting losses")
        plot_losses(plot_location)

    if output_file == 'stdout':
        print("Outputting results to stdout\n\n\n")
        print(json.dumps(all_anomalies))
    else:
        if uses_different_indexes:
            output_file = output_file[0:-5] + '.part.' + str(start_distr) + '.' + str(end_distr) + '.json'
        print("Outputting results to", output_file)
        with open(output_file, 'w') as out_file:
            out_file.write(json.dumps(all_anomalies))

    print("Done, closing files and stuff")
    try:
        sys.exit()
    except AttributeError:
        print("Tensorflow threw some error while closing, just ignore it")


if __name__ == "__main__":
    main()
