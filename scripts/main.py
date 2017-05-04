"""The main script file"""
import sys, getopt
import pandas as pd
import tensorflow as tf
import features as features

LSTM_SIZE = 2 ** 4
inputs = tf.placeholder(tf.float32, [None, 32, 32, 3])
BATCH_SIZE = tf.shape(inputs[0])[0]
REPORTING_SIZE = 1000
SPEED_REPORTING_SIZE = 1000
ENABLE_REPORTING = False
MAX_ROWS = 5000

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
    def __init__(self, enable):
        self.time_slices = list()
        self.start_time = time.time()
        self.last_slice_time = self.start_time
        self.enable = enable

    def do_slice(self, label):
        if self.enable:
            slice_dict = dict()
            slice_dict["name"] = label
            slice_dict["time"] = time.time() - self.last_slice_time
            self.last_slice_time = time.time()
            self.time_slices.append(slice_dict)


def convert_to_tensor(data):
    """Converts given data to a tensor"""
    return tf.convert_to_tensor(data, name="auth_small")






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
            output, self._state = self.model(
                tf.convert_to_tensor(self.queue, dtype=tf.float32, name="Features"), self.state)

            self._last_output = output

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


def get_io_settings(argv):
    """This gets the input and output files from the command line arguments"""
    input_file = '/data/s1481096/LosAlamos/data/auth_small.h5'
    output_file = sys.stdout

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

def iterate(argv):
    """Iterates over the data and feeds it to the RNN"""

    input_file, output_file = get_io_settings(argv)

    time_measurer_container = TimeMeasureContainer()
    global rows
    rows = 0

    model = init_model()

    for name, group in pd.read_hdf('/data/s1481096/LosAlamos/data/auth_small.h5', 'auth_small')\
            .groupby(["source_user"], start=0, stop=MAX_ROWS):
        for row in group.itertuples():

            time_measurer = TimeMeasure(ENABLE_REPORTING)
            handle_row(model, Row(row), time_measurer)

            time_measurer_container.append(time_measurer)

            if rows % SPEED_REPORTING_SIZE == 1:
                print("At row", rows, "total time is", time.time() - start_time,
                      "so total rows per second is", rows / (time.time() - start_time))

            if rows % REPORTING_SIZE == 1:
                time_measurer_container.generate_report()

    return model


def main():
    """The main function"""
    iterate()



if __name__ == "__main__":
    main(sys.argv[1:])
