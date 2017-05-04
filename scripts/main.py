"""The main script file"""
import numpy as np
import tensorflow as tf
import features as features
import sys, getopt, json, pickle

LSTM_SIZE = 2 ** 4
inputs = tf.placeholder(tf.float32, [None, 32, 32, 3])
BATCH_SIZE = tf.shape(inputs[0])[0]
SPEED_REPORTING_SIZE = 1000
rows = 0

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
    """An RNN"""

    def __init__(self, scope, input_, is_training_model=False, num_steps=20, data_size=None):
        self.scope = scope
        with tf.name_scope(scope):
            lstm = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE, state_is_tuple=True)
            state = lstm.zero_state(BATCH_SIZE, tf.float32)

        self.is_training = is_training_model

        self._model = lstm
        self._final_state = state
        self._initial_state = state
        self._input = input_

        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = lstm(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, LSTM_SIZE])
        # softmax_w = tf.get_variable(
        #     "softmax_w", [LSTM_SIZE, data_size], dtype=tf.float32)
        # softmax_b = tf.get_variable("softmax_b", [data_size], dtype=tf.float32)
        # logits = tf.matmul(output, softmax_w) + softmax_b
        # loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        #     [logits],
        #     [tf.reshape(input_.targets, [-1])],
        #     [tf.ones([BATCH_SIZE * num_steps], dtype=tf.float32)])
        # self._cost = cost = tf.reduce_sum(loss) / BATCH_SIZE
        # self._final_state = state
        #
        # if not is_training:
        #     return
        #
        self._lr = tf.Variable(0.0, trainable=False)
        # tvars = tf.trainable_variables()
        # grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
        #                                   config.max_grad_norm)
        # optimizer = tf.train.GradientDescentOptimizer(self._lr)
        # self._train_op = optimizer.apply_gradients(
        #     zip(grads, tvars),
        #     global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    @property
    def lr(self):
        return self._lr

    def assign_lr(self, session, lr_value):
        """Assigns the learning rate for this model"""
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def run_epoch(self, session, eval_op=None):
        costs = 0.0
        iters = 0
        state = session.run(self._initial_state)

        fetches = {
            "final_state": self._final_state,
        }
        if eval_op is not None:
            fetches["eval_op"] = eval_op

        for step in range(self._input.epoch_size):
            feed_dict = {}
            for i, (c, h) in enumerate(self._initial_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h

            vals = session.run(fetches, feed_dict)
            cost = vals["cost"]
            state = vals["final_state"]

            costs += cost
            iters += self._input.num_steps

        return np.exp(costs / iters)



class UserNetwork:
    """The class describing a single model and all its corresponding data"""

    def __init__(self, data, epochs=10, lr_decay=0.5, learning_rate=1.0):
        """Creates a new set of networks"""

        self.user_name = data["user_name"]
        self.dataset = data["datasets"]

        self.models = {
            "training": Model("training", self.dataset["training"],
                              is_training_model=True, data_size=len(self.dataset["training"])),
            "validation": Model("validation", self.dataset["validation"],
                                data_size=len(self.dataset["validation"])),
            "test": Model("test", self.dataset["test"],
                          data_size=len(self.dataset["test"]))
        }

        self.config = {
            "epochs": epochs,
            "lr_decay": lr_decay,
            "learning_rate": learning_rate
        }

    def find_anomalies(self):
        sv = tf.train.SuperVisor()
        with sv.managed_session() as session:
            for i in range(self.config["epochs"]):
                lr_decay = self.config["lr_decay"] ** max(i + 1 - self.config["epochs"], 0.0)
                self.models["training"].assign_lr(session, self.config["learning_rate"] * lr_decay)

                session.run(self.models["training"].lr)
                self.models["training"].run_epoch(session)
                self.models["validation"].run_epoch(session)

            return self.models["test"].run_epoch(session)


def find_anomalies(data):
    """Finds anomalies in given data
    
    data format: {
        "user_name": str,
        "datasets": {
            "training": list(int/float[8]),
            "validation": list(int/float[8]),
            "test": list(int/float[8])
        }
    }
    """
    network = UserNetwork(data)
    anomalies = network.find_anomalies()
    return anomalies


def get_io_settings(argv):
    """This gets the input and output files from the command line arguments"""
    input_file = '/data/s1495674/features.p'
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

def main(argv):
    """The main function"""

    input_file, output_file = get_io_settings(argv)

    with open(input_file, 'rb') as in_file:
        users_list = pickle.load(in_file)

    print("Starting training")

    all_anomalies = list()

    for user in users_list:
        anomalies = find_anomalies(user)
        if len(anomalies) > 0:
            all_anomalies.append(anomalies)

    print(all_anomalies)



if __name__ == "__main__":
    main(sys.argv[1:])
