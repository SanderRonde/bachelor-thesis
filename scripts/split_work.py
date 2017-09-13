import os
import sys
import time
import glob
import json
import getopt
from typing import List, Dict, Tuple, Union
import matplotlib.pyplot as plt
from string import Template
import numpy as np
from imports.log import logline_to_folder, enter_group, exit_group
from imports.timer import Timer
import traceback

plt.switch_backend('agg')

# Global variables
SPLIT_WINDOW_TEMPLATE = Template('tmux split-window -t $target -$orientation $command')
SWAP_PANES_TEMPLATE = Template('tmux swap-pane -d -D')
SELECT_PANE_TEMPLATE = Template('tmux select-pane -t $pane')
ADD_QUOTES = Template("'$str'")
VERTICAL = 'v'
HORIZONTAL = 'h'
DEBUG = False
LOG = False
REMOVE_INPUT_FILES = False
EXIT_ON_PLOT_ERROR = False
SKIP_JOINING = False
SKIP_MAIN = False


def get_command_template() -> Template:
    if DEBUG:
        return Template("echo \"cuda=$cuda_device, start=$start, end=$end\" && CUDA_VISIBLE_DEVICES=$cuda_device "
                        "$command -s $start -d $end; sleep 10")
    return Template("echo Activating source... && (conda info --envs | grep \"*\" | grep \"nn\" || source activate "
                    "nn) > /dev/null ; "
                    " echo \"Starting...\" && CUDA_VISIBLE_DEVICES=$cuda_device $command -s $start -d $end; sleep 10")


def recalc_gpus(start: int, amount: int, available: List[int]) -> List[int]:
    used = [(x + start) for x in range(amount)]
    return list(filter(lambda x: x in available, used))


def run_cmd(command: str) -> int:
    if DEBUG or LOG:
        logline(command)
    return os.system(command)


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


def get_io() -> Tuple[GPUSource, str, str, str]:
    argv = sys.argv[1:]

    gpu_amount = 16
    command = 'python3 main.py'
    name = 'nn'
    logfile = None

    available_gpus = [x for x in range(gpu_amount)]

    gpu_offset = 0
    to_use_gpus = recalc_gpus(gpu_offset, gpu_amount, available_gpus)

    try:
        opts, args = getopt.getopt(argv, 'g:c:n:s:l:a:hds')
    except getopt.GetoptError:
        error('Command line arguments invalid')
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
            to_use_gpus = recalc_gpus(gpu_offset, gpu_amount, available_gpus)
        elif opt == '-d':
            global DEBUG
            DEBUG = True
        elif opt == '-l':
            logfile = arg
        elif opt == '-h':
            print('Options:')
            print(' -g <gpu_amount>		The amount of GPUs amongst which to split work')
            print(' -c <command>		The command to run')
            print(' -n <name>           The name of the tmux window')
            print(' -s <start_index>    The index of the first GPU to use')
            print(' -a <available_gpus> All GPUs that are available to use (comma separated)')
            print(' -l <logfile>        Log executed commands')
            print(' -d                  Enable debug mode')
            sys.exit()
        else:
            print('Unrecognized argument passed, refer to -h for help')
            sys.exit(2)

    return GPUSource(to_use_gpus), command, name, logfile


def gen_command_str(command: str, indexes: Tuple[int, int, int]) -> Union[None, str]:
    if indexes[0] == -1 and indexes[1] == -1:
        # Empty pane, no command
        return None
    return ADD_QUOTES.substitute(str=get_command_template().substitute(cuda_device=indexes[2],
                                                                       command=command, start=indexes[0],
                                                                       end=indexes[1]))


def split_pane(command: str, indexes: Tuple[int, int, int], pane_index: int, orientation: str):
    cmd = gen_command_str(command, indexes)
    if cmd is not None:
        return_code = run_cmd(SPLIT_WINDOW_TEMPLATE.substitute(target=pane_index, orientation=orientation,
                                                               command=cmd))
        if return_code == 256:
            # Pane too small
            error("Could not create panel")


def calc_distribution(gpus: GPUSource) -> Dict[int, Tuple[int, int, int]]:
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


def join_output(out_file: str):
    # Go to out_file directory
    split_dir = out_file.split('/')
    os.chdir('/'.join(split_dir[0:-1]))

    file_name = split_dir[-1]
    glob_pattern = file_name[0:-5] + '.part.*.json'
    files = glob.glob(glob_pattern)
    files.sort(key=lambda name: int(name.split('.')[-3]))

    files_length = len(files)
    with open(out_file, 'w') as output:
        output.write('[')
        for i in range(files_length):
            file = files[i]
            with open(file) as infile:
                for line in infile:
                    output.write(line)
            if i != files_length - 1:
                output.write(',')
        output.write(']')

    logline("Combined outputs and wrote them to", out_file)

    if REMOVE_INPUT_FILES:
        # Remove the partial files
        for file in files:
            os.remove(file)

        logline("Removed partial files")
    else:
        logline("Skipping removal of partial files")


def split_plots_by_name(names: List[str]):
    try:
        current_name = names[0].split('.')[-5]
        current_list = list()
        for i in range(len(names)):
            if names[i].split('.')[-5] == current_name:
                current_list.append(names[i])
            else:
                yield current_list
                current_list = list()
                current_name = names[i].split('.')[-5]
    except IndexError as err:
        error(err)
        error("Something crashed the main.py files")
        if EXIT_ON_PLOT_ERROR:
            sys.exit(1)


def plot_arr(values: List[float], scalar: float = 1.0, normalize=False):
    x_vals = list()
    y_vals = list()

    if normalize:
        values = [float(i) / max(values) for i in values]

    for i in range(len(values)):
        x_vals.append(i * scalar)
        y_vals.append(values[i])

    plt.plot(x_vals, y_vals, 'ro', markersize=1)


def recalc_highest_offenders(data: Dict[str, Dict[str, float]], max_highest_offenders: int) -> Dict[str, float]:
    data_list = list()

    avg_list = list()
    for key, val in data.items():
        if key == 'avg.':
            avg_list = val
        else:
            try:
                data_list.append({
                    "key": key,
                    "val": val["val"],
                    "sorting_value": val["sorting_value"]
                })
            except TypeError as err:
                error('An invalid value was found', key, val)
                raise err

    if len(avg_list) > 0:
        if type(avg_list[0]) == list:
            avg_list = list(map(lambda x: x[0], avg_list))
        data_list.append({
            "key": "avg.",
            "val": sum(avg_list) / len(avg_list),
            # Sorting value as low as possible to have it be at the end
            "sorting_value": 0.0
        })

    data_list.sort(key=lambda x: x["sorting_value"], reverse=False)
    shortened_list = data_list[-max_highest_offenders:]

    shortened_dict = dict()
    for i in range(len(shortened_list)):
        shortened_dict[shortened_list[i]["key"]] = shortened_list[i]["val"]

    return shortened_dict


def do_dict_plot(metadata: Dict[str, Union[str, bool]],
                 data: Union[
                     List[Union[float, List[float]]], Dict[str, Union[str, float]], Dict[str, Dict[str, float]]]):
    fig, axes = plt.subplots()

    has_avg = False
    x_vals = list()
    y_vals = list()
    for key, val in data.items():
        if key == 'avg.':
            has_avg = True
            continue
        x_vals.append(key)
        y_vals.append(val)

    if has_avg:
        x_vals.append('avg.')
        y_vals.append(data['avg.'])

    if len(y_vals) == 0:
        error('Values are empty, plotting empty figure')
        axes.plot([], markersize=1)
    else:
        if metadata["is_box_plot"]:
            axes.boxplot(y_vals, 'rs', vert=True, patch_artist=True)
        else:
            axes.plot(y_vals, 'ro')

        plt.setp(axes, xticks=[y + 1 for y in range(len(x_vals))],
                 xticklabels=x_vals)


def do_plot(metadata: Dict[str, Union[str, bool]],
            data: Union[List[Union[float, List[float]]], Dict[str, Union[str, float]], Dict[str, Dict[str, float]]],
            fig_idx: int):
    if metadata['is_highest_offenders']:
        data = recalc_highest_offenders(data, metadata['max_highest_offenders'])

    if metadata['is_sorted']:
        data = sorted(data)

    plt.figure(fig_idx)
    plt.subplot(111)

    enter_group()
    logline("Gathering data points for plot", metadata["name"] + ('...' if metadata["name"] == 'losses' else ''))
    if metadata["is_dict"]:
        do_dict_plot(metadata, data)
    elif metadata['normalize_x']:
        biggest_sample_size = 0
        for i in range(len(data)):
            if len(data[i]) > biggest_sample_size:
                biggest_sample_size = len(data[i])

        for i in range(len(data)):
            sample_size = len(data[i])
            scalar = biggest_sample_size / max(sample_size - 1, 1)

            plot_arr(data[i], scalar=scalar, normalize=metadata['normalize_y'])
    else:
        if metadata['multidimensional']:
            for i in range(len(data)):
                plot_arr(data[i], normalize=metadata['normalize_y'])
        else:
            plt.plot(np.arange(0, len(data), 1.0), data, 'ro', markersize=1)

    if metadata['is_log']:
        plt.yscale('log')
    plt.ylabel(metadata['y_label'])
    plt.xlabel(metadata['x_label'])
    plt.savefig(metadata['plot_location'] + metadata['name'] + '.png', dpi=1500)
    logline("Saved plot to", metadata['plot_location'] + metadata['name'] + '.png')


def handle_dataset(metadata: Dict[str, Union[str, bool]],
                   data: Union[
                       List[Union[float, List[float]]], Dict[str, Union[str, float]], Dict[str, Dict[str, float]]],
                   fig_idx: int):
    if metadata is None:
        error('Skipping plot because metadata is empty')
        return

    if metadata['is_plot']:
        do_plot(metadata, data, fig_idx)
    else:
        logline('Skipping plotting as this is a data-only "plot"')

    with open(metadata['plot_location'] + metadata['name'] + '.json', 'w') as json_out:
        json_out.write(json.dumps(data))

    logline('Saved JSON output to', metadata['plot_location'] + metadata['name'] + '.json')
    exit_group()

    logline("")


def join_plots(plot_dir: str):
    os.chdir(plot_dir)

    enter_group()
    logline('Joining plots')

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)

    glob_pattern = '*.part.*.json'
    files = glob.glob(glob_pattern)
    files.sort(key=lambda name: name.split('.')[-5])

    figs = 1
    if len(files) == 0:
        error('No partial files were written')
        sys.exit(1)

    for plots_for_name in split_plots_by_name(files):
        plots_for_name.sort(key=lambda name: int(name.split('.')[-3]))

        files_length = len(plots_for_name)
        plotting_metadata = None
        data_parts = None
        for i in range(files_length):
            file = plots_for_name[i]
            with open(file) as plot_data_part:
                try:
                    data_part = json.loads(plot_data_part.read())
                except json.decoder.JSONDecodeError:
                    error("File contents of", file, "are empty, skipping it")
                    continue
                if plotting_metadata is None:
                    plotting_metadata = {
                        "name": data_part['name'],
                        "is_log": data_part['is_log'],
                        "x_label": data_part['x_label'],
                        "y_label": data_part['y_label'],
                        "is_plot": data_part['is_plot'],
                        "is_dict": data_part["is_dict"],
                        "is_sorted": data_part['is_sorted'],
                        "is_box_plot": data_part["is_box_plot"],
                        "normalize_x": data_part['normalize_x'],
                        "normalize_y": data_part['normalize_y'],
                        "plot_location": data_part['plot_location'],
                        "multidimensional": data_part['multidimensional'],
                        "is_highest_offenders": data_part['is_highest_offenders'],
                        "max_highest_offenders": data_part['max_highest_offenders']
                    }
                if data_parts is None:
                    if plotting_metadata["is_dict"]:
                        data_parts = dict()
                    else:
                        data_parts = list()

                if plotting_metadata["is_dict"]:
                    for key, val in data_part['data'].items():
                        if key == 'avg.':
                            if 'avg.' in data_parts:
                                avg_arr = data_parts['avg.']
                            else:
                                avg_arr = []
                            avg_arr.append(val)
                            data_parts['avg.'] = avg_arr
                        else:
                            data_parts[key] = val
                else:
                    data_parts = data_parts + data_part['data']

        handle_dataset(plotting_metadata, data_parts, figs)
        figs = figs + 1

    logline("Done with plots")
    exit_group()


def join_files(cmd: str):
    # Get output file from command input
    parts = cmd.split(' ')

    out_file = '/data/s1495674/anomalies.encoded.json'
    plot_dir = '/data/s1495674/plot_data/'
    for i in range(len(parts)):
        if parts[i] == '-o':
            out_file = parts[i + 1]
        if parts[i] == '-p':
            plot_dir = parts[i + 1]

    join_output(out_file)
    join_plots(plot_dir)

def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True

def main(gpus: GPUSource, command: str, name: str):
    assert not SKIP_MAIN or not SKIP_JOINING, 'Script should do something'

    if SKIP_MAIN:
        logline('Skipping main process')
    else:
        distribution = calc_distribution(gpus)

        pids = list()

        if gpus.length > 8:
            # Do 16
            run_cmd('tmux new-session -d ' + gen_command_str(command,
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

            for i in range(gpus.length - 16):
                distr_index = i + 16

                indexes = get_indexes_at_new_position(distr_index, distribution)
                cmd = gen_command_str(command, indexes)
                if cmd is not None:
                    pid = os.spawnl(os.P_NOWAIT, cmd)
                    pids.append(pid)
        else:
            # Do 8 only
            run_cmd('tmux new-session -d ' + gen_command_str(command,
                                                             get_indexes_at_new_position(0, distribution)))
            split_pane(command, get_indexes_at_new_position(4, distribution), 0, 'v')
            split_pane(command, get_indexes_at_new_position(2, distribution), 0, 'v')
            split_pane(command, get_indexes_at_new_position(6, distribution), 1, 'v')

            split_pane(command, get_indexes_at_new_position(1, distribution), 0, 'h')
            split_pane(command, get_indexes_at_new_position(5, distribution), 1, 'h')
            split_pane(command, get_indexes_at_new_position(3, distribution), 2, 'h')
            split_pane(command, get_indexes_at_new_position(7, distribution), 3, 'h')

        run_cmd('tmux new-window ' + ADD_QUOTES.substitute(str=name))
        get_eta_cmd = "tmux capture-pane -pS -150 | tac | grep -e 'ETA is' | head -1 |" \
                      " egrep -o 'ETA is \([0-9]+h\)?\([0-9]+m\)?[0-9]+s' || echo 'ETA is Unknown'"
        get_percentage_cmd = "tmux capture-pane -pS -150 | tac | grep -e 'Checking user' | head -1 |" \
                             " egrep -o '[0-9]+%' || echo '0%'"
        run_cmd(
            Template('tmux set-option status-left "#[fg=#00ba00]#($GET_ETA_CMD) - #($GET_PERCENTAGE_CMD)"').substitute(
                GET_ETA_CMD=get_eta_cmd,
                GET_PERCENTAGE_CMD=get_percentage_cmd
            ))
        run_cmd('tmux -2 attach-session -d')

        while len(pids) > 0:
            time.sleep(30)
            pids = list(filter(lambda x: pid_is_running(x), pids))
            debug(len(pids), 'processes still running')

    global main_done
    main_done = time.time()

    if SKIP_JOINING:
        logline('Skipping joining of files')
    else:
        try:
            if not SKIP_MAIN:
                # There's no point in waiting seeing as this is the only point of running
                logline('About to join files, press CTRL+C to cancel')
                time.sleep(20)
                logline('Last chance...')
                time.sleep(20)
            logline('Joining files...')
            enter_group()
            join_files(command)
            exit_group()
            logline("Done joining all files")
        except KeyboardInterrupt as _:
            logline('Cancelled joining of files')


if __name__ == '__main__':
    _gpus, _command, _name, _logfile = get_io()

    logline, debug, error, log_done = logline_to_folder(path=_logfile)
    start_time = time.time()
    main_done = None

    exit_code = 0
    try:
        main(_gpus, _command, _name)
    except Exception as e:
        error("An exception has occurred", "\n",
              traceback.format_exc())
        exit_code = 1
    else:
        logline('Ran successfully')
    finally:
        logline('Runtime for training/testing is', Timer.stringify_time(Timer.format_time(main_done - start_time)))
        logline('Total runtime is', Timer.stringify_time(Timer.format_time(time.time() - start_time)))
        log_done()
        sys.exit(exit_code)
