import os
import sys
import getopt
from typing import Tuple
from string import Template


def get_io() -> Tuple[int, str, str]:
    argv = sys.argv[1:]

    gpu_amount = 16
    command = 'python3 main.py'
    name = 'nn'

    try:
        opts, args = getopt.getopt(argv, 'g:c:n:h')
    except getopt.GetoptError:
        print('Command line arguments invalid')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-g':
            gpu_amount = int(arg)
        elif opt == '-c':
            command = arg
        elif opt == '-n':
            name = arg
        elif opt == '-h':
            print('Options:')
            print(' -g <gpu_amount>		The amount of GPUs amongst which to split work')
            print(' -c <command>		The command to run')
            print(' -n <name>           The name of the tmux window')
            sys.exit()
        else:
            print('Unrecognized argument passed, refer to -h for help')
            sys.exit(2)

    return gpu_amount, command, name


COMMAND_TEMPLATE = Template("$command -s $start -d $end")
ADD_QUOTES = Template("'$str'")


def gen_command_str(command: str, start: int, end: int) -> str:
    return ADD_QUOTES.substitute(str=command)
    # return ADD_QUOTES.substitute(str=COMMAND_TEMPLATE.substitute(command=command, start=start, end=end))


def split_pane(command: str, start: int, end: int, is_horizontal: bool):
    exit_code = os.system('tmux split-window -' + ('h ' if is_horizontal else 'v ') +
                          gen_command_str(command, start, end))
    if exit_code == 256:
        # Pane too small
        print("Bad")


def main():
    gpu_amount, command, name = get_io()

    increment = 100.0 / float(gpu_amount)
    current_index = increment

    print(gen_command_str(command, 0, round(current_index) - 1))
    os.system('tmux new-session -d ' + gen_command_str(command, 0, round(current_index) - 1))
    is_horizontal = True
    for i in range(gpu_amount - 1):
        start_index = round(current_index)
        end_index = round(start_index + increment) - 1

        split_pane(command, start_index, end_index, is_horizontal)

        is_horizontal = not is_horizontal

        current_index += increment

    os.system('tmux new-window ' + ADD_QUOTES.substitute(str=name))
    os.system('tmux -2 attach-session -d')


if __name__ == '__main__':
    main()
