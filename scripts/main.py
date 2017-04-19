"""The main script file"""
import pandas as pd
import tensorflow as tf
import features as features
from datetime import datetime, date, time

LSTM_SIZE = 2 ** 4
BATCH_SIZE = 1
CHUNKSIZE = 2 ** 10
MODELS = dict()


def convert_to_tensor(data):
    """Converts given data to a tensor"""
    return tf.convert_to_tensor(data, name="auth_small")


class Row:
    """A row of data"""
    def __init__(self, row):
        row_one_split = row[1].split("@")
        row_two_split = row[2].split("@")

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


class Features:
    """All the features fr a model"""

    _domains = list()
    _dest_users = list()
    _src_computers = list()
    _dest_computers = list()

    def __init__(self, row):
        self._last_access = row.time

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


def get_valid_scope_name(name):
    """Turns given name string into a valid scope name for tensorflow"""
    return name.replace('$', '_').lower().replace(' ', '_')


class Model:
    """The class describing a single model and all its corresponding data"""

    def __init__(self, row):
        self._scope = get_valid_scope_name(row.user)
        with tf.variable_scope(self._scope):
            """Creates a new Model for user with given name"""

            lstm = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE, state_is_tuple=False)
            state_ = lstm.zero_state(BATCH_SIZE, tf.float32)

            self._name = row.user
            self._model = lstm
            self._state = state_
            self._last_output = None
            self._features = Features(row)
            self._runs = 0

    @property
    def model(self):
        """The TensorFlow model itself"""
        return self._model

    @property
    def state(self):
        """The TensorFlow state"""
        return self._state

    @property
    def name(self):
        """The name of the user"""
        return self._name

    @property
    def output(self):
        """The last output of the RNN"""
        return self._last_output

    @property
    def features(self):
        """The features of this model"""
        return self._features

    @property
    def runs(self):
        """The amount of times this model has been executed (trained)"""
        return self._runs

    def update_features(self, row):
        """Updates the features after a run"""
        self.features.update_dest_users(row)
        
    def run(self, data):
        """Runs one instance of the RNN with given data as input"""

        with tf.variable_scope(self._scope, reuse=(self.runs > 0)):
            self._features.update(data)
            data = features.extract(data, self)
            self._last_output, self._state = self.model(data, self.state)
        self._runs += 1


def assert_model_in_dict(row):
    """Makes sure there is a model for given user in the dictionary"""
    if row.user not in MODELS:
        print("Creating another model for user " + row.user)
        MODELS[row.user] = Model(row)


def do_iteration(user, row):
    """Does one iteration of the RNN with given data as input"""
    MODELS[user].run(row)


def handle_row(row):
    """Handles one row of the original data"""
    if row.user == "ANONYMOUS LOGON":
        return

    assert_model_in_dict(row)
    do_iteration(row.user, row)


def iterate():
    """Iterates over the data and feeds it to the RNN"""
    for chunk in pd.read_hdf('/data/s1481096/LosAlamos/data/auth_small.h5', 'auth_small',
                             chunksize=CHUNKSIZE):
        for name, group in chunk.groupby(
                [chunk.index, pd.TimeGrouper(freq='Min')]):
            for row in group.itertuples():
                handle_row(Row(row))


def main():
    """The main function"""
    iterate()

if __name__ == "__main__":
    main()