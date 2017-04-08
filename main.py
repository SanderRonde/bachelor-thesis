import pandas as pd
import tensorflow as tf

lstm_size = 2 ** 7
chunksize = 2 ** 8

models = dict()

def getRowUser(row):
    return row[3]

def generateTFModel():
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    print("Chunk size is", chunksize, "state size is", lstm_size) 
    state = tf.zeros([chunksize, lstm_size])
    print("Zeros is", state)
    print("Type of zeros is", type(state))
    return lstm, state

def assertModelInDict(user):
    if user not in models:
        print("Creating another model for user" + user)
        lstm, state = generateTFModel()
        models[user] = [lstm, state]

def doIteration(user, data):
    print(models)
    print(models[user])
    print(models[user][0])
    print(models[user][1])
    lstm = models[user][0]
    oldState = models[user][1]
    print(type(lstm))
    print(type(oldState))
    output, newState = lstm(data, tf.zeros([chunksize, lstm_size]))
    print("Output is", output)
    models[user] = [lstm, newState]

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
