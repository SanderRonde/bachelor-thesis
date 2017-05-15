"""The main script file"""
import getopt
import os
import pickle
import sys
import time
import math
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
    output_file = '/data/s1495674/anomalies.txt'
    plot_location = None
    start_distr = 0
    end_distr = 100

    try:
        opts, args = getopt.getopt(argv, 'i:o:p:e:d:s:khvx')
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
        elif opt == '-s':
            start_distr = int(arg)
        elif opt == '-x':
            global VERBOSE_RUNNING
            VERBOSE_RUNNING = False
        elif opt == '-h':
            print("Options:")
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

LOSSES = list()

Dataset = Dict[str, Union[str, Dict[str, List[List[float]]]]]


class FeatureDescriptor:
    def __init__(self, name: str, type_: str, weight: float):
        self.name = name
        self.type = type_
        self.weight = weight


FEATURE_MAP = [FeatureDescriptor("time_since_last_access", "number", 1.0),
               FeatureDescriptor("unique_domains", "number", 1.0),
               FeatureDescriptor("unique_dest_users", "number", 1.0),
               FeatureDescriptor("unique_src_computers", "number", 1.0),
               FeatureDescriptor("unique_dest_computers", "number", 1.0),
               FeatureDescriptor("most_freq_src_computer", "number", 1.0),
               FeatureDescriptor("most_freq_dest_computer", "number", 1.0),
               FeatureDescriptor("auth_type", "nominal", 1.0),
               FeatureDescriptor("logon_type", "nominal", 1.0),
               FeatureDescriptor("auth_orientation", "nominal", 1.0),
               FeatureDescriptor("success_failure", "binary", 1.0),
               FeatureDescriptor("percentage_failed_logins", "number", 5.0)]

FEATURE_SIZE = len(FEATURE_MAP)
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


class FeatureDeviation:
    def __init__(self, deviation: float, name: str, predicted: float, actual: float):
        self.deviation = deviation
        self.name = name
        self.predicted = predicted
        self.actual = actual

    def anti_normalize(self, maxes: float):
        self.deviation *= maxes
        self.predicted *= maxes
        self.actual *= maxes


class Anomaly:
    def __init__(self, index: int):
        self.context = None
        self._index = index
        self._list = list()

    def append(self, deviation: FeatureDeviation):
        self._list.append(deviation)

    def get_total(self) -> float:
        anomaly_score = 0.0
        for i in range(len(self._list)):
            anomaly_score += self._list[i].deviation

        return anomaly_score

    def anti_normalize(self, maxes: List[float]):
        for i in range(len(self._list)):
            self._list[i].anti_normalize(maxes[i])

        context_arr = np.array(self.context)
        reshaped = np.reshape(context_arr, (context_arr.shape[1], context_arr.shape[0]))
        for col in range(len(reshaped)):
            reshaped[col] = [float(i) * maxes[col] for i in reshaped[col]]
        self.context = np.reshape(reshaped, (context_arr.shape[0], context_arr.shape[1]))

    @property
    def list(self) -> List[FeatureDeviation]:
        return self._list

    def add_context(self, dataset: List[List[float]]):
        self.context = dataset


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
    def prepare_data(training_data: List[List[float]], test_data: List[List[float]]):
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

    def fit(self, train_x, train_y, epochs: int = 10) -> list:
        """Fits the model to given training data"""

        train_history = list()

        for i in range(epochs):
            global VERBOSE_RUNNING
            if VERBOSE_RUNNING:
                print("Epoch", i, '/', epochs)
            verbosity = 0
            if VERBOSE_RUNNING:
                VERBOSE_RUNNING = 1
            train_history.append(self.model.fit(train_x, train_y, batch_size=BATCH_SIZE,
                                                epochs=1, verbose=verbosity, shuffle=False))
            if not GIVE_TEST_SET_PREVIOUS_KNOWLEDGE or i != epochs - 1:
                self.model.reset_states()

        return train_history

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


class UserNetwork:
    """The class describing a single model and all its corresponding data"""

    def __init__(self, model: RNNModel, data: Dataset, epochs: int = 10):
        """Creates a new set of networks"""

        self.user_name = data["user_name"]
        self.dataset = data["datasets"]
        self.config = {
            "epochs": epochs
        }

        self.model = model
        self.model.reset()

    @staticmethod
    def _get_deviation_for_feature(feature: float, actual: float, descriptor: FeatureDescriptor):
        if descriptor.type == "number":
            # Simple number relation, use ratio
            return FeatureDeviation(descriptor.weight * abs_ratio(feature, actual),
                                    descriptor.name, feature, actual)
        if descriptor.type == "nominal":
            # Nominal relationship, if it's not within range of the number, mark as 1,
            # otherwise mark as the difference
            diff = abs(actual - feature)
            return FeatureDeviation(min(diff, 1.0) * descriptor.weight,
                                    descriptor.name, feature, actual)
        if descriptor.type == "binary":
            # Binary relationship,
            return FeatureDeviation(abs(actual - feature) * descriptor.weight,
                                    descriptor.name, feature, actual)

    def _is_anomaly(self, predicted: List[float], actual: List[float], index: int) -> Union[bool, Anomaly]:
        possible_anomaly = Anomaly(index)
        for i in range(len(predicted)):
            possible_anomaly.append(UserNetwork._get_deviation_for_feature(predicted[i], actual[i], FEATURE_MAP[i]))

        if possible_anomaly.get_total() >= ANOMALY_THRESHOLD:
            # Only if it's an anomaly, attach context in order to avoid memory leaks
            if GIVE_TEST_SET_PREVIOUS_KNOWLEDGE:
                context = (self.dataset["training"] + self.dataset["test"])[
                          len(self.dataset["training"]) + (index - CONTEXT_LENGTH):len(self.dataset["train"]) + index]
            else:
                context = self.dataset["test"][index - CONTEXT_LENGTH:index]
            possible_anomaly.add_context(context)
            return possible_anomaly
        return False

    def find_anomalies(self) -> List[Anomaly]:
        # Train the network first
        train_x, train_y, test_x, test_y = RNNModel.prepare_data(self.dataset["training"], self.dataset["test"])
        history = self.model.fit(train_x, train_y, epochs=self.config["epochs"])
        last_loss = history[-1].history["loss"][0]

        print("\nChecking losses on test set...")
        losses = self.model.test(test_x, test_y)
        print("Done checking losses on test set\n")

        anomalies = list()

        global LOSSES
        losses_list = list()

        for i in range(len(losses)):
            losses_list.append(last_loss / losses[i])
            # possible_anomaly = self._is_anomaly(predictions[i], test_y[i], i)
            # if possible_anomaly:
            #     anomalies.append(possible_anomaly)

        LOSSES.append(losses_list)
        return anomalies


def find_anomalies(model: RNNModel, data: Dataset) -> List[Anomaly]:
    """Finds anomalies in given data
    
    data format: {
        "user_name": str,
        "weights": list(float[12]),
        "datasets": {
            "training": list(float[12]),
            "test": list(float[12])
        }
    }
    """
    network = UserNetwork(model, data, epochs=EPOCHS)
    anomalies = network.find_anomalies()
    return anomalies


def format_anomaly(anomaly: Anomaly, maxes: List[float]) -> str:
    # Return all context rows first
    return_str = ""
    if VERBOSE:
        last_str = ""
        for i in range(len(anomaly.context)):
            last_str = str(anomaly.context[i]) + "\n"
            return_str += last_str

        last_line_length = len(last_str) - 1
        for i in range(last_line_length):
            return_str += "^"

        return_str += "\n\n"
    else:
        return_str += "\n"

    anomaly.anti_normalize(maxes)
    features = anomaly.list

    if VERBOSE:
        for i in range(len(features)):
            return_str += "Feature " + features[i].name + " was predicted as " + ("%.5f" % features[i].predicted) + \
                          " and was " + ("%.5f" % features[i].actual) + " giving a total deviation of " + \
                          ("%.5f" % features[i].deviation) + "\n"
        return_str += "\nTotal anomaly score is: " + ("%.5f" % anomaly.get_total()) + " and maximum is " + \
                      ("%.5f" % ANOMALY_THRESHOLD)
    else:
        return_str += str(list(map(lambda x: x.actual, features)))

    return return_str


def format_anomalies(anomalies: List[Dict[str, Union[Dataset, List[Anomaly]]]]) -> str:
    output = ""
    for i in range(len(anomalies)):
        user_anomalies = anomalies[i]
        maxes = user_anomalies["user"]["maxes"]
        if VERBOSE:
            output += "Anomalies for user " + user_anomalies["user"]["user_name"] + "\n"
        for j in range(len(user_anomalies["anomalies"])):
            single_anomaly = user_anomalies["anomalies"][j]
            output += format_anomaly(single_anomaly, maxes)

        output += "\n\n"
    return output


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
    users_list, uses_different_indexes = get_user_list(full_list, start_distr, end_distr)
    print("There are", total_users, "users, and this process is doing", len(users_list), "of them")

    try:
        model = RNNModel()
    except tf.errors.InternalError:
        print("No GPU is currently available for you, aborting")
        raise

    print("Starting anomaly detection")

    all_anomalies = list()

    total_samples = 0
    for user in users_list:
        total_samples += len(user["datasets"]["training"])

    timer = Timer(total_samples)

    tested_users = 0
    for user in users_list:

        if tested_users > 0:
            print("\nChecking user", tested_users, "/", total_users, "ETA is " + timer.get_eta())

        try:
            anomalies = find_anomalies(model, user)
            if len(anomalies) > 0:
                all_anomalies.append({
                    "user": user,
                    "anomalies": anomalies
                })
            tested_users += 1
            timer.add_to_current(len(user["datasets"]["training"]))
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
        print(format_anomalies(all_anomalies))
    else:
        if uses_different_indexes:
            output_file = output_file[0:-4] + '.part.' + str(start_distr) + '.' + str(end_distr) + '.txt'
        print("Outputting results to", output_file)
        with open(output_file, 'w') as out_file:
            out_file.write('Testing for distr ' + str(start_distr) + ' to ' + str(end_distr))
            #out_file.write(format_anomalies(all_anomalies))

    print("Done, closing files and stuff")
    try:
        sys.exit()
    except AttributeError:
        print("Tensorflow threw some error while closing, just ignore it")


if __name__ == "__main__":
    main()
