import datetime
import sys


def logline(*args, spaces_between=True, end_line=True):
    # Get the current time
    sys.stdout.write(datetime.datetime.now().strftime('%H:%M:%S | '))
    if spaces_between:
        print(*args)
    else:
        for word in args:
            sys.stdout.write(word)
        if end_line:
            sys.stdout.write('\n')

