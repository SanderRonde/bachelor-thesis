import datetime
import sys
from typing import List

DEBUG_COLOR_PREFIX = '\x1b[0;37;42m'
ERROR_COLOR_PREFIX = '\x1b[0;37;41m'
COLOR_POSTFIX = '\x1b[0m'

group_length = 0


def print_prefix(output=sys.stdout, debug_mode=False, error_mode=False, prefix=None,
                 indent=True):
    if prefix is None:
        prefix = group_length
    if group_length > 0:
        # print('prefix length is', prefix)
        data_prefix = ((prefix - 1) * ' ') + ('|' if indent else '')
    else:
        data_prefix = ''
    if indent:
        data_prefix = data_prefix + ' '
    if debug_mode:
        time_str = '%H:%M:%S D'
    elif error_mode:
        time_str = '%H:%M:%S E'
    else:
        time_str = '%H:%M:%S |'
    output.write(datetime.datetime.now().strftime(time_str + data_prefix))
    return True


def logline(*args, output=sys.stdout, spaces_between: bool = True, end_line: bool = True,
            debug_mode: bool = False, error_mode: bool = False, indent=True):
    # Get the current time
    print_prefix(output=output, debug_mode=debug_mode, error_mode=error_mode, indent=indent)
    is_first = True

    prefix = DEBUG_COLOR_PREFIX if debug_mode else ERROR_COLOR_PREFIX if error_mode else ''
    postfix = COLOR_POSTFIX if debug_mode or error_mode else ''
    for word in args:
        if not is_first and spaces_between:
            output.write(prefix + ' ' + postfix)
        output.write(prefix + str(word) + postfix)

        is_first = False
    if end_line:
        output.write('\n')


def debug(*args, output=sys.stdout, spaces_between: bool = True, end_line: bool = True):
    logline(*args, output=output, spaces_between=spaces_between, end_line=end_line, debug_mode=True)


def error(*args, output=sys.stdout, spaces_between: bool = True, end_line: bool = True):
    logline(*args, output=output, spaces_between=spaces_between, end_line=end_line, error_mode=True)


def enter_group():
    logline('\\', indent=False)
    global group_length
    group_length = group_length + 1


def exit_group():
    global group_length
    group_length = group_length - 1
    if group_length < 0:
        group_length = 0
        error('Attempting to reduce groups even though you\'re already at root')
    logline('/', indent=False)


def logline_proxy(log_file, *args, spaces_between: bool = True, end_line: bool = True,
                  is_debug: bool = False, is_error: bool = False):
    logline(*args, spaces_between=spaces_between, end_line=end_line,
            output=log_file, debug_mode=is_debug, error_mode=is_error)
    logline(*args, spaces_between=spaces_between, end_line=end_line,
            output=sys.stdout, debug_mode=is_debug, error_mode=is_error)


def gen_proxy(log_file, is_debug=False, is_error=False):
    return lambda *args, spaces_between=True, end_line=True: logline_proxy(
        log_file, *args, spaces_between=spaces_between, end_line=end_line,
        is_debug=is_debug, is_error=is_error
    )


def close_logs_file(file):
    print_prefix(prefix=0)
    print('Finishing logs...')
    file.write('\nEnd of log entry\n')
    file.close()
    print_prefix(prefix=0)
    print('Done writing logs')


def logline_to_folder(folder_loc: str = None, file_name: str = None, path: str = None, start: int = 0, end: int = 100):
    if (folder_loc is None or file_name is None) and path is None:
        return logline, debug, error, lambda: None
    else:
        if folder_loc:
            if not folder_loc.endswith('/'):
                folder_loc = folder_loc + '/'

            path = folder_loc + 'main_logs.log'

        if start is None:
            start = 0
        if end is None:
            end = 100
        if start != 0 or end != 100:
            path = folder_loc + 'main_logs' + '.' + str(start) + '.' + str(end) + '.log'

        file = open(path, 'a+')

        file.write('\n\n\nNew Log Entry\n\n')
        return gen_proxy(file), gen_proxy(file, is_debug=True), gen_proxy(file, is_error=True), \
               lambda: close_logs_file(file)
