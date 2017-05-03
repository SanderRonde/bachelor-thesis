"""The main script file"""
import pandas as pd
import tensorflow as tf
import features as features
import time

LSTM_SIZE = 2 ** 4
inputs = tf.placeholder(tf.float32, [None, 32, 32, 3])
BATCH_SIZE = tf.shape(inputs[0])[0]
CHUNKSIZE = 2 ** 10
MAX_QUEUE_LENGTH = 2500
REPORTING_SIZE = 10000
SPEED_REPORTING_SIZE = 2500

rows = 0
start_time = time.time()


class TimeMeasureContainer:
    def __init__(self):
        self.parts = list()
        self.megalist = list()

    def append(self, measurer):
        self.parts.append(measurer)

    def append_slice_to_megalist(self, index, name, duration):
        if len(self.megalist) <= index:
            slice_dict = dict()
            slice_dict["name"] = name
            slice_dict["time"] = -1 * duration
            self.megalist.append(slice_dict)

        self.megalist[index]["time"] = self.megalist[index].get("time") + duration

    def generate_megalist(self):
        for data_part in self.parts:
            for label_index in range(len(data_part.time_slices)):
                self.append_slice_to_megalist(label_index,
                                              data_part.time_slices[label_index].get("name"),
                                              data_part.time_slices[label_index].get("time"))

    def generate_report(self):
        self.generate_megalist()
        for index in range(len(self.megalist)):
            duration = self.megalist[index].get("time")
            avg_time = duration
            if index is 0:
                print("Time between slice start and " + self.megalist[index].get("name") + " is %.2f" % avg_time)
            else:
                print("Time between slice " + self.megalist[index - 1].get("name") + " and " +
                      self.megalist[index].get("name") + " is %.2f" % avg_time)

        del self.parts[:]


class TimeMeasure:
    def __init__(self):
        self.time_slices = list()
        self.start_time = time.time()
        self.last_slice_time = self.start_time

    def do_slice(self, label):
        return
        slice_dict = dict()
        slice_dict["name"] = label
        slice_dict["time"] = time.time() - self.last_slice_time
        self.last_slice_time = time.time()
        self.time_slices.append(slice_dict)


def convert_to_tensor(data):
    """Converts given data to a tensor"""
    return tf.convert_to_tensor(data, name="auth_small")


class SpecialDict(dict):
    def lol(self):
        return 'lol'

class SpecialList(list):
    def lol(self):
        return 'lol'


class Row:
    """A row of data"""

    def __init__(self, row):
        row_one_split = row[1].split("@")
        row_two_split = row[2].split("@")

        self._row = row
        self.time = row[0]
        self.source_user = self.user = row_one_split[0]
        self.domain = row_one_split[1]
        self.dest_user = row_two_split[0]
        self.src_computer = row[3]
        self.dest_computer = row[4]
        self.auth_type = row[5]
        self.logon_type = row[6]
        self.auth_orientation = row[7]
        self.status = row[8]

    def to_str(self):
        return self._row


class Features:
    """All the features fr a model"""

    def __init__(self, row):
        self._last_access = row.time
        self._domains = list()
        self._dest_users = list()
        self._src_computers = list()
        self._dest_computers = list()

    def update_dest_users(self, user):
        """Updates the dest_users list"""
        if user != "?" and user not in self._dest_users:
            self._dest_users.append(user)

    def update_src_computers(self, computer):
        """Updates the src_computers list"""
        if computer != "?" and computer not in self._src_computers:
            self._src_computers.append(computer)

    def update_dest_computers(self, computer):
        """Updates the dest_computers list"""
        if computer != "?" and computer not in self._dest_computers:
            self._dest_computers.append(computer)

    def update_domains(self, domain):
        """Updates the dest_users list"""
        if domain != "?" and domain not in self._domains:
            self._domains.append(domain)

    def update(self, row):
        """Updates all data lists for this feature class"""
        self.update_dest_users(row.dest_user)
        self.update_src_computers(row.src_computer)
        self.update_dest_computers(row.dest_computer)
        self.update_domains(row.domain)
        self._last_access = row.time

    @property
    def last_access(self):
        """The last time this user has authenticated themselves"""
        return self._last_access

    @property
    def dest_users(self):
        """All destination users"""
        return self._dest_users

    @property
    def src_computers(self):
        """All source computers"""
        return self._src_computers

    @property
    def dest_computers(self):
        """All destination computers"""
        return self._dest_computers

    @property
    def domains(self):
        """All domains accessed"""
        return self._domains


class User:
    """A user in the dataset"""

    def __init__(self, row):
        """Creates a new user"""

        self.name = row.user
        self.features = Features(row)

    def get_features(self, data, time_measurer):
        """Prepares one row of data for feeding into the model"""

        self.features.update(data)
        time_measurer.do_slice("Updating features")
        data = features.extract(data, self, time_measurer)
        time_measurer.do_slice("Converting to tensor")

        return data


def get_valid_scope_name(name):
    """Turns given name string into a valid scope name for tensorflow"""
    return name.replace('$', '_').lower().replace(' ', '_')


class Model:
    """The class describing a single model and all its corresponding data"""

    def __init__(self):
        """Creates a new Model"""

        lstm = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE, state_is_tuple=True)
        state_ = lstm.zero_state(BATCH_SIZE, tf.float32)

        self._model = lstm
        self._state = state_
        self._last_output = None
        self.queue = list()
        self.current_user = None

    @property
    def model(self):
        """The TensorFlow model itself"""
        return self._model

    @property
    def state(self):
        """The TensorFlow state"""
        return self._state

    @property
    def output(self):
        """The last output of the RNN"""
        return self._last_output

    def update_user(self, user):
        self.current_user = user

    def get_current_user_name(self):
        if self.current_user is not None:
            return self.current_user.name
        return "ANONYMOUS LOGON"

    def run_if_switching_user(self, row, time_measurer):
        """Runs the RNN for the previous user when a switch is made to a new one"""
        if row.user != self.get_current_user_name():
            time_measurer.do_slice("About to run")
            if self.current_user is not None:
                self.run()
            time_measurer.do_slice("Ran")
            self.current_user = User(row)

    def run(self):
        """Runs the RNN with its current queue as input"""
        print("About to run model for user", self.get_current_user_name(),
              " with a queue size of", len(self.queue))
        with tf.variable_scope(get_valid_scope_name(self.get_current_user_name())):
            self._last_output, self._state = self.model(
                tf.convert_to_tensor(self.queue, dtype=tf.float32, name="Features"), self.state)

        del self.queue[:]
        self._state = self.model.zero_state(BATCH_SIZE, tf.float32)


def append_data_to_queue(model, row, time_measurer):
    """Appends the data for this row to the queue"""
    model.queue.append(model.current_user.get_features(row, time_measurer))


def handle_row(model, row, time_measurer):
    """Handles one row of the original data"""
    if row.user == "ANONYMOUS LOGON":
        return

    global rows
    rows += 1

    model.run_if_switching_user(row, time_measurer)
    time_measurer.do_slice("Appending data to queue")
    append_data_to_queue(model, row, time_measurer)


def init_model():
    """Initializes the RNN"""
    return Model()


def iterate():
    """Iterates over the data and feeds it to the RNN"""

    time_measurer_container = TimeMeasureContainer()
    global rows
    rows = 0

    model = init_model()

    for name, group in pd.read_hdf('/data/s1481096/LosAlamos/data/auth_small.h5', 'auth_small')\
            .groupby(["source_user"]):
        for row in group.itertuples():

            time_measurer = TimeMeasure()
            handle_row(model, Row(row), time_measurer)

            time_measurer_container.append(time_measurer)

            if rows % SPEED_REPORTING_SIZE == 1:
                print("At row", rows, "total time is", time.time() - start_time,
                      "so total rows per second is", rows / (time.time() - start_time))

            if rows % REPORTING_SIZE == 1 and False:
                time_measurer_container.generate_report()


def main():
    """The main function"""
    iterate()


if __name__ == "__main__":
    main()
