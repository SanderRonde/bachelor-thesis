import os
import sys
import json
from imports.log import logline, debug, error
from imports.io import IO, IOInput
from typing import List, Dict, Tuple, Union
from string import Template


io = IO({
    'f': IOInput(None, str, arg_name='folder',
                 descr='The folder in which to store it all',
                 alias='folder')
})


def ensure_trailing_slash(folder_name: str) -> str:
    """Makes sure a string ends with a slash"""
    if not folder_name.endswith('/'):
        return folder_name + '/'
    return folder_name


def ensure_folder(folder_name: str):
    """Checks if a folder exists and if it doesn't, makes it"""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def gen_folders():
    """Generates all folders needed for the process"""
    root_folder = ensure_trailing_slash(io.get('folder'))
    ensure_folder(root_folder)
    ensure_folder(root_folder + 'logs/')
    ensure_folder(root_folder + 'plots/')


def main():
    gen_folders()


if __name__ == "__main__":
    logline('Starting folder generation')
    main()
    logline('Done generating folders')
