import pandas as pd
import datetime

def iterate():
    chunksize = 10 ** 6
    for chunk in pd.read_hdf('/data/s1481096/LosAlamos/data/auth_small.h5', 'auth',
            chunksize=chunksize):
        for name, group in chunk.groupby(
                [chunk.index, pd.TimeGrouper(freq='Min')]):
            for row in group.itertuples():
                print(row)

def main():
    iterate()

if __name__ == "__main__":
    main()