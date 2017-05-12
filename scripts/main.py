"""The main script file"""
import os
import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM
import sys, getopt, pickle, time, math, os
from typing import List, Dict, Tuple, Union
from keras.layers.core import Dense, Activation, Dropout
np.random.seed(1234)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LSTM_SIZE = 2 ** 4
BATCH_SIZE = 32
SPEED_REPORTING_SIZE = 1000
ANOMALY_THRESHOLD = 1.0
CONTEXT_LENGTH = 10
EPOCHS = 25
GIVE_TEST_SET_PREVIOUS_KNOWLEDGE = True
VERBOSE = True

Dataset = Dict[str, Union[str, Dict[str, List[List[float]]]]]

class FeatureDescriptor:
    def __init__(self, name: str, type_: str, weight: float):
        self.name = name
        self.type = type_
        self.weight = weight

FEATURE_MAP = [FeatureDescriptor("time_since_last_access", "number", 1.0),
<<<<<<< HEAD
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

    def get_eta(self) -> int:
        passed_time = time.time() - self.start_time
        amount_done = self._current / self._max

        return math.round(((1 / amount_done) * passed_time) - passed_time)

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
            reshaped[col] = [float(i)*maxes[col] for i in reshaped[col]]
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


    def fit(self, train_x, train_y, epochs:int=10):
        """Fits the model to given training data"""
        for i in range(epochs):
            print("Epoch", i, '/', epochs)
            self.model.fit(train_x, train_y, batch_size=BATCH_SIZE,
                           epochs=1, verbose=1, shuffle=False)
            if not GIVE_TEST_SET_PREVIOUS_KNOWLEDGE or i != epochs - 1:
                self.model.reset_states()

    def predict(self, data) -> list:
        """Predicts the result for given test data"""
        return self.model.predict(data)


def abs_ratio(a: float, b: float) -> float:
    if b == 0:
        return 1000
    ratio = a / b
    if ratio < 1:
        return 1 / ratio
    return ratio

class UserNetwork:
    """The class describing a single model and all its corresponding data"""

    def __init__(self, model: RNNModel, data: Dataset, epochs:int=10):
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
                    len(self.dataset["training"]) + (index - CONTEXT_LENGTH):len(self.dataset["train"] + index)
                ]
            else:
                context = self.dataset["test"][index-CONTEXT_LENGTH:index]
            possible_anomaly.add_context(context)
            return possible_anomaly
        return False


    def find_anomalies(self) -> List[Anomaly]:
        # Train the network first
        train_x, train_y, test_x, test_y = RNNModel.prepare_data(self.dataset["training"], self.dataset["test"])
        self.model.fit(train_x, train_y, epochs=self.config["epochs"])

        # Do prediction on test set
        predictions = self.model.predict(test_x)

        anomalies = list()

        for i in range(len(predictions)):
            possible_anomaly = self._is_anomaly(predictions[i], test_y[i], i)
            if possible_anomaly:
                anomalies.append(possible_anomaly)
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


def get_io_settings(argv: sys.argv) -> Tuple[str, str]:
    """This gets the input and output files from the command line arguments"""
    input_file = '/data/s1495674/features.p'
    output_file = '/data/s1495674/anomalies.txt'

    try:
        opts, args = getopt.getopt(argv, 'i:o:')
    except getopt.GetoptError:
        print("Command line arguments invalid")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-i':
            input_file = arg
        elif opt == '-o':
            output_file = arg
        else:
            print("Unrecognized argument passed")
            sys.exit(2)

    return input_file, output_file


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
            return_str += "Feature " + features[i].name + " was predicted as " + ("%.5f" % features[i].predicted) +\
                          " and was " + ("%.5f" % features[i].actual) + " giving a total deviation of " + \
                          ("%.5f" % features[i].deviation) + "\n"
        return_str += "\nTotal anomaly score is: " + ("%.5f" % anomaly.get_total()) + " and maximum is " + \
                          ("%.5f" %ANOMALY_THRESHOLD)
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


def main(argv: sys.argv):
    """The main function"""

    input_file, output_file = get_io_settings(argv)

    with open(input_file, 'rb') as in_file:
        users_list = pickle.load(in_file)

    total_users = len(users_list)
    print("There are", total_users, "users")

    print("Compiling generic model")
    model = RNNModel()

    print("Starting anomaly detection")

    all_anomalies = list()

    total_samples = 0
    for user in users_list:
        total_samples += len(user["datasets"]["training"])
    timer = Timer(total_samples)

    tested_users = 0
    for user in users_list:

        print("\nChecking user", tested_users, "/", total_users, "ETA is" + str(timer.get_eta()) + "s")

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
            print("Skipping rest of the users")
            break

    print("Done checking users, outputting results now")

    if output_file is sys.stdout:
        print("Outputting results to stdout\n\n\n")
        print(format_anomalies(all_anomalies))
    else:
        print("Outputting results to", output_file)
        with open(output_file, 'w') as out_file:
            out_file.write(format_anomalies(all_anomalies))

    print("Done, closing files and stuff")
    try:
        sys.exit()
    except AttributeError:
        print("Tensorflow threw some error while closing, just ignore it")


if __name__ == "__main__":
    main(sys.argv[1:])
