import pandas as pd
import tensorflow as tf

lstm_size = 100
chunksize = 2 ** 8

models = dict()

def getRowUser(row):
    return row[3]

def generateTFModel():
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    state = tf.zeros([chunksize, lstm.state_size])
    return [lstm, state]

def assertModelInDict(user):
    if user not in models:
        models[user] = generateTFModel()

def doIteration(user, data):
    models[user] = models[user][0](data, models[user][1])

def handleRow(row):
    user = getRowUser(row)
    assertModelInDict(user)
    doIteration(user, row)

def iterate():
    for chunk in pd.read_hdf('/data/s1481096/LosAlamos/data/auth_small.h5', 'auth_small',
            chunksize=chunksize):
        for name, group in chunk.groupby(
                [chunk.index, pd.TimeGrouper(freq='Min')]):
            for row in group.itertuples():
                handleRow(row)            

def main():
    iterate()

if __name__ == "__main__":
    main()
