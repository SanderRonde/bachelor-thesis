import datetime
import sys
from typing import List


def print_time(output=sys.stdout):
    output.write(datetime.datetime.now().strftime('%H:%M:%S | '))
    return True


def logline(*args, output=sys.stdout, spaces_between: bool = True, end_line: bool = True):
    # Get the current time
    print_time(output=output)
    if spaces_between:
        orig_out = sys.stdout
        sys.stdout = output
        print(*args)
        sys.stdout = orig_out
    else:
        for word in args:
            output.write(str(word))
        if end_line:
            output.write('\n')


def logline_proxy(log_file, *args, spaces_between: bool = True, end_line: bool = True):
    logline(*args, spaces_between=spaces_between, end_line=end_line,
            output=log_file)
    logline(*args, spaces_between=spaces_between, end_line=end_line,
            output=sys.stdout)


def close_logs_file(file):
    print_time()
    print('Finishing logs...')
    file.write('\nEnd of log entry')
    file.close()
    print_time()
    print('Done writing logs')


def logline_to_folder(folder_loc: str, start: int, end: int):
    if folder_loc is None:
        return logline, lambda : None
    else:
        if not folder_loc.endswith('/'):
            folder_loc = folder_loc + '/'

        file_name = folder_loc + 'main_logs.log'
        if start != 0 or end != 100:
            file_name = folder_loc + 'main_logs' + '.' + str(start) + '.' + str(end) + '.log'

        file = open(file_name, 'a+')

        file.write('\n\n\nNew Log Entry\n\n')
        return lambda *args, spaces_between = True, end_line = True: \
                   logline_proxy(file, *args, spaces_between=spaces_between,
                                 end_line=end_line), \
               lambda : close_logs_file(file)