"""The script that generates all 3 datasets and their features"""
import pandas as pd

MAX_ROWS = 50000

users_list = list()

def main():
	"""The main function"""
	for name, group in pd.read_hdf('/data/s1481096/LosAlamos/data/auth_small.h5', 'auth_small')\
            .groupby(["source_user"], start=0, stop=MAX_ROWS):
		if group.


if __name__ == "__main__":
    main()