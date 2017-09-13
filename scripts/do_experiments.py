import os
import sys
import json
import time
from imports.log import logline, debug, error
from imports.io import IO, IOInput
from typing import List, Dict, Tuple, Union, Callable, Any
import socket


SLEEP_TIME = 20
EXPERIMENT_NOT_DONE = 0
EXPERIMENT_BUSY = 1
EXPERIMENT_DONE = 2


io = IO({
    'i': IOInput(None, str, arg_name='experiments_file',
                 descr='The path to the experiments file',
                 alias='experiments_file')
})


def get_experiments(experiments_file_location: str) -> Dict[str, int]:
    with open(experiments_file_location, 'r+') as experiments_file:
        try:
            obj = json.loads(experiments_file.read())
        except Exception:
            # No state set yet
            set_state(experiments_file_location, 0)
            return 0

        if obj["error"]:
            error('An error occurred in another instance, exiting with error code', obj['error_code'])
            sys.exit(obj['error_code'])
        return obj


def upload_experiments(file_location: str, experiments: Dict[str, int], is_error=False, error_code=0):
    with open(file_location, 'w+') as experiments_file:
        if is_error:
            logline('Set state to error state with code', error_code)
            experiments["error"] = error_code
        else:
            logline('Set experiments file to to', experiments)
        experiments_file.write(json.dumps(experiments))


def do_job(job: str, state_file: str = None) -> int:
    if state_file is not None:
        return os.system(job + ' -s ' + state_file)
    return os.system(job)


def do_first_job(file_location: str):
    experiments = get_experiments(file_location)

    did_job = False
    for cmd, state in experiments.items():
        if state == EXPERIMENT_NOT_DONE:
            # Do this job
            experiments[cmd] = EXPERIMENT_BUSY
            upload_experiments(file_location, experiments)
            did_job = True

            exit_code = do_job(cmd)
            if exit_code != 0:
                error('An error occurred while executing command', cmd, 'giving exit code', exit_code)
                upload_experiments(file_location, experiments, is_error=True, error_code=exit_code)
            else:
                experiments[cmd] = EXPERIMENT_DONE
                logline('Done with job', cmd)
                upload_experiments(file_location, experiments)
            break

    return did_job


def main():
    experiments_file_location = io.get('experiments_file')

    if experiments_file_location is None:
        logline('Experiment file is not specified, please do so')
        sys.exit(2)

    logline('Waiting', SLEEP_TIME * 2, 'seconds to allow for setup...')
    time.sleep(SLEEP_TIME * 2)
    logline('Starting')

    try:
        while do_first_job(experiments_file_location):
            logline('Did job')
        logline('Completed successfully!')
    except KeyboardInterrupt:
        logline('Cancelling run due to user interrupt')
        sys.exit(130)
    logline('Done')

if __name__ == "__main__":
    main()
