import os
import sys
import glob
import getopt
from typing import List, Dict, Tuple
from string import Template


SPLIT_WINDOW_TEMPLATE = Template('tmux split-window -t $target -$orientation $command')
SWAP_PANES_TEMPLATE = Template('tmux swap-pane -d -D')
SELECT_PANE_TEMPLATE = Template('tmux select-pane -t $pane')
ADD_QUOTES = Template("'$str'")
VERTICAL = 'v'
HORIZONTAL = 'h'
DEBUG = False


def get_command_template() -> Template:
    if DEBUG:
        return Template("echo \"cuda=$cuda_device, start=$start, end=$end\" && CUDA_VISIBLE_DEVICES=$cuda_device $command")
    return Template("CUDA_VISIBLE_DEVICES=$cuda_device $command -s $start -d $end")


def recalc_gpus(start: int, amount: int, available: List[int]) -> List[int]:
    used = [(x + start) for x in range(amount)]
    return list(filter(lambda x: x in available, used))


def run_cmd(command: str):
    if DEBUG:
        print(command)
    os.system(command)


class GPUSource:
    def __init__(self, gpus: List[int]):
        self._gpus = gpus
        self._length = len(gpus)

    def get(self):
        item = self._gpus[0]
        self._gpus = self._gpus[1:]
        return item

    @property
    def length(self):
        return self._length


def get_io() -> Tuple[GPUSource, str, str]:
    argv = sys.argv[1:]

    gpu_amount = 16
    command = 'python3 main.py -x'
    name = 'nn'

    available_gpus = [x for x in range(gpu_amount)]

    gpu_offset = 0
    to_use_gpus = recalc_gpus(gpu_offset, gpu_amount, available_gpus)

    try:
        opts, args = getopt.getopt(argv, 'g:c:n:s:a:hd')
    except getopt.GetoptError:
        print('Command line arguments invalid')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-g':
            gpu_amount = int(arg)
            to_use_gpus = recalc_gpus(gpu_offset, gpu_amount, available_gpus)
        elif opt == '-c':
            command = arg
        elif opt == '-n':
            name = arg
        elif opt == '-s':
            gpu_offset = int(arg)
            to_use_gpus = recalc_gpus(gpu_offset, gpu_amount, available_gpus)
        elif opt == '-a':
            available_gpus = list(map(lambda x: int(x), arg.split(',')))
            print(available_gpus)
            to_use_gpus = recalc_gpus(gpu_offset, gpu_amount, available_gpus)
        elif opt == '-d':
            global DEBUG
            DEBUG = True
        elif opt == '-h':
            print('Options:')
            print(' -g <gpu_amount>		The amount of GPUs amongst which to split work')
            print(' -c <command>		The command to run')
            print(' -n <name>           The name of the tmux window')
            print(' -s <start_index>    The index of the first GPU to use')
            print( '-a <available_gpus> All GPUs that are available to use (comma separated)')
            print(' -d                  Enable debug mode')
            sys.exit()
        else:
            print('Unrecognized argument passed, refer to -h for help')
            sys.exit(2)

    return GPUSource(to_use_gpus), command, name


def gen_command_str(command: str, indexes: Tuple[int, int, int]) -> str:
    if indexes[0] == -1 and indexes[1] == -1:
        # Empty pane, no command
        return ''
    return ADD_QUOTES.substitute(str=get_command_template().substitute(cuda_device= indexes[2],
                                                                 command=command, start=indexes[0], end=indexes[1]))


def split_pane(command: str, indexes: Tuple[int, int, int], pane_index: int, orientation: str):
    exit_code = run_cmd(SPLIT_WINDOW_TEMPLATE.substitute(target=pane_index, orientation=orientation,
                                                           command=gen_command_str(command, indexes)))
    if exit_code == 256:
        # Pane too small
        print("Could not create panel")


def calc_distribution(gpus: GPUSource) -> Dict[int, Tuple[int, int, int]]:
    print(gpus._gpus)
    gpu_amount = gpus.length

    distribution = dict()

    increment = 100.0 / float(gpu_amount)
    current_index = increment

    distribution[0] = (0, round(current_index) - 1, gpus.get())

    for i in range(gpu_amount - 2):
        start_index = round(current_index)
        end_index = round(float(start_index) + increment) - 1

        distribution[i + 1] = (start_index, end_index, gpus.get())

        current_index += increment

    distribution[gpu_amount - 1] = (round(current_index), 100, gpus.get())

    return distribution


def get_indexes_at_new_position(new_position: int, distr: Dict[int, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if new_position in distr:
        return distr[new_position]
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
    gpus, command, name = get_io()

    distribution = calc_distribution(gpus)

    if gpus.length > 8:
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