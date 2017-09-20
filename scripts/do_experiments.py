import os
import sys
import json
import time
from string import Template
from imports.log import logline, debug, error
from imports.io import IO, IOInput
from typing import List, Dict, Tuple, Union, Callable, Any
import socket


SLEEP_TIME = 20
EXPERIMENT_NOT_DONE = 0
EXPERIMENT_BUSY = 1
EXPERIMENT_DONE = 2
NOTIFY_TEMPLATE = Template('~/scripts/scripts/notify.sh "$message"')


io = IO({
    'i': IOInput(None, str, arg_name='experiments_file',
                 descr='The path to the experiments file',
                 alias='experiments_file'),
    'g': IOInput(False, bool, arg_name='use_gpus',
                 descr='Whether to use GPUs',
                 alias='use_gpus',
                 has_input=False)
})


def get_experiments(experiments_file_location: str) -> Dict[str, int]:
    with open(experiments_file_location, 'r+') as experiments_file:
        try:
            obj = json.loads(experiments_file.read())
        except Exception:
            # No state set yet
            set_state(experiments_file_location, 0)
            return {}

        if 'error' in obj:
            error('An error occurred in another instance, exiting with error code', obj['error_code'])
            sys.exit(obj['error_code'])
        return obj


def upload_experiments(file_location: str, experiments: Dict[str, int], is_error=False, error_code=0):
    with open(file_location, 'w+') as experiments_file:
        if is_error:
            logline('Set state to error state with code', error_code)
            experiments["error_code"] = error_code
        else:
            logline('Updated experiments file')
        experiments_file.write(json.dumps(experiments))


def do_job(job: str) -> int:
    return os.system(job)


def try_notify(message: str):
    try:
        do_job(NOTIFY_TEMPLATE.substitute(message=message))
    except Exception as e:
        error('Got an error notifying that job is done, np', e)


def do_first_job(file_location: str):
    experiments = get_experiments(file_location)

    did_job = False
    idx = 0
    for cmd, state in experiments.items():
        idx = idx + 1
        if state == EXPERIMENT_NOT_DONE:
            # Do this job
            experiments[cmd] = EXPERIMENT_BUSY
            upload_experiments(file_location, experiments)
            did_job = True

            exit_code = do_job(cmd + (' -g 16' if io.get('use_gpus') else ' -g 0'))
            if exit_code != 0:
                error('An error occurred while executing command', cmd, 'giving exit code', exit_code)
                try_notify('A command failed')
                upload_experiments(file_location, experiments, is_error=True, error_code=exit_code)
                sys.exit(1)
            else:
                experiments[cmd] = EXPERIMENT_DONE
                logline('Done with job', cmd)
                try_notify('Done with job ' + str(idx))
                upload_experiments(file_location, experiments)
            break

    return did_job


def main():
    experiments_file_location = io.get('experiments_file')

    if experiments_file_location is None:
        logline('Experiment file is not specified, please do so')
        sys.exit(2)

    logline('Starting')

    try:
        logline('Starting jobs')
        while do_first_job(experiments_file_location):
            logline('Did job')
        logline('Completed successfully!')
    except KeyboardInterrupt:
        logline('Cancelling run due to user interrupt')
        sys.exit(130)
    logline('Done')

if __name__ == "__main__":
    main()
