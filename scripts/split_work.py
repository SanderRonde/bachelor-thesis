import os
import sys
import glob
import getopt
from typing import List, Dict, Tuple, Union, TypeVar
from string import Template


GPU_OFFSET = 0


COMMAND_TEMPLATE = Template("CUDA_VISIBLE_DEVICES=$cuda_device $command -s $start -d $end")
SPLIT_WINDOW_TEMPLATE = Template('tmux split-window -t $target -$orientation $command')
SWAP_PANES_TEMPLATE = Template('tmux swap-pane -d -D')
SELECT_PANE_TEMPLATE = Template('tmux select-pane -t $pane')
ADD_QUOTES = Template("'$str'")
VERTICAL = 'v'
HORIZONTAL = 'h'


def get_io() -> Tuple[int, str, str]:
    argv = sys.argv[1:]

    gpu_amount = 16
    command = 'python3 main.py -x'
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
        elif opt == '-s':
            global GPU_OFFSET
            GPU_OFFSET = int(arg)
        elif opt == '-h':
            print('Options:')
            print(' -g <gpu_amount>		The amount of GPUs amongst which to split work')
            print(' -c <command>		The command to run')
            print(' -n <name>           The name of the tmux window')
            print(' -s <start_index>    The index of the first GPU to use')
            sys.exit()
        else:
            print('Unrecognized argument passed, refer to -h for help')
            sys.exit(2)

    return gpu_amount, command, name


def gen_command_str(command: str, indexes: Tuple[int, int, int]) -> str:
    if indexes[0] == -1 and indexes[1] == -1:
        # Empty pane, no command
        return ''
    return ADD_QUOTES.substitute(str=COMMAND_TEMPLATE.substitute(cuda_device= GPU_OFFSET + indexes[2],
                                                                 command=command, start=indexes[0], end=indexes[1]))


def split_pane(command: str, indexes: Tuple[int, int, int], pane_index: int, orientation: str):
    exit_code = os.system(SPLIT_WINDOW_TEMPLATE.substitute(target=pane_index, orientation=orientation,
                                                           command=gen_command_str(command, indexes)))
    if exit_code == 256:
        # Pane too small
        print("Could not create panel")


def calc_distribution(gpu_amount: int) -> Dict[int, Tuple[int, int]]:
    distribution = dict()

    increment = 100.0 / float(gpu_amount)
    current_index = increment

    distribution[0] = (0, round(current_index) - 1)

    for i in range(gpu_amount - 2):
        start_index = round(current_index)
        end_index = round(float(start_index) + increment) - 1

        distribution[i + 1] = (start_index, end_index)

        current_index += increment

    distribution[gpu_amount - 1] = (round(current_index), 100)

    return distribution


def get_indexes_at_new_position(new_position: int, distr: Dict[int, Tuple[int, int]]) -> Tuple[int, int, int]:
    if new_position in distr:
        return distr[new_position][0], distr[new_position][1], new_position
    return -1, -1, new_position


def join_files(command: str):
    print("Joining output files")

    # Get output file from command input
    parts = command.split(' ')

    out_file = '/data/s1495674/anomalies.txt'
    for i in range(len(parts)):
        if parts[i] == '-o':
            out_file = parts[i + 1]
            break

    # Go to out_file directory
    split_dir = out_file.split('/')
    os.chdir('/'.join(split_dir[0:-1]))

    file_name = split_dir[-1]
    glob_pattern = file_name[0:-4] + '.part.*.txt'
    files = glob.glob(glob_pattern)
    files.sort(key=lambda name: int(name.split('.')[-3]))

    with open(out_file, 'w') as output:
        for file in files:
            with open(file) as infile:
                for line in infile:
                    output.write(line)

    print("Combined outputs and wrote them to", out_file)

    # Remove the partial files
    for file in files:
        os.remove(file)

    print("Removed partial files")


def main():
    gpu_amount, command, name = get_io()

    distribution = calc_distribution(gpu_amount)

    if gpu_amount > 8:
        # Do 16
        os.system('tmux new-session -d ' + gen_command_str(command,
                                                           get_indexes_at_new_position(0, distribution)))
        split_pane(command, get_indexes_at_new_position(8, distribution), 0, 'v')
        split_pane(command, get_indexes_at_new_position(4, distribution), 0, 'v')
        split_pane(command, get_indexes_at_new_position(12, distribution), 1, 'v')

        split_pane(command, get_indexes_at_new_position(2, distribution), 0, 'h')
        split_pane(command, get_indexes_at_new_position(10, distribution), 1, 'h')
        split_pane(command, get_indexes_at_new_position(6, distribution), 2, 'h')
        split_pane(command, get_indexes_at_new_position(14, distribution), 3, 'h')

        split_pane(command, get_indexes_at_new_position(1, distribution), 0, 'h')
        split_pane(command, get_indexes_at_new_position(9, distribution), 1, 'h')
        split_pane(command, get_indexes_at_new_position(5, distribution), 2, 'h')
        split_pane(command, get_indexes_at_new_position(13, distribution), 3, 'h')
        split_pane(command, get_indexes_at_new_position(3, distribution), 4, 'h')
        split_pane(command, get_indexes_at_new_position(11, distribution), 5, 'h')
        split_pane(command, get_indexes_at_new_position(7, distribution), 6, 'h')
        split_pane(command, get_indexes_at_new_position(15, distribution), 7, 'h')
    else:
        # Do 8 only
        os.system('tmux new-session -d ' + gen_command_str(command,
                                                           get_indexes_at_new_position(0, distribution)))
        split_pane(command, get_indexes_at_new_position(4, distribution), 0, 'v')
        split_pane(command, get_indexes_at_new_position(2, distribution), 0, 'v')
        split_pane(command, get_indexes_at_new_position(6, distribution), 1, 'v')

        split_pane(command, get_indexes_at_new_position(1, distribution), 0, 'h')
        split_pane(command, get_indexes_at_new_position(5, distribution), 1, 'h')
        split_pane(command, get_indexes_at_new_position(3, distribution), 2, 'h')
        split_pane(command, get_indexes_at_new_position(7, distribution), 3, 'h')


    os.system('tmux new-window ' + ADD_QUOTES.substitute(str=name))
    os.system('tmux -2 attach-session -d')

    join_files(command)


if __name__ == '__main__':
    main()
