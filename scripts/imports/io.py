import sys
import getopt
from typing import Dict, Union, Callable, Any, Generic, TypeVar, Tuple, List


T = TypeVar('T')


class IOInput(Generic[T]):
    def __init__(self, default_value: T, data_type: Union[type, Callable[[str], T]],
                 has_input=True, arg_name=None, descr=None, alias=None):
        self.default_value = default_value
        self.data_type = data_type
        self.value = default_value

        self._has_input = has_input
        self.arg_name = arg_name
        self.descr = descr
        self.alias = alias

    def update(self, arg: str):
        if self.data_type == str:
            self.value = arg
        elif self.data_type == int:
            self.value = int(arg)
        elif self.data_type == float:
            self.value = float(arg)
        elif self.data_type == bool:
            self.value = not self.value
        else:
            self.value = self.data_type(arg)

    def gen_help(self, key: str) -> Tuple[str, str]:
        help_str = ' -' + key
        if self._has_input:
            help_str += ' <' + self.arg_name + '>'
        return help_str, self.descr

    @property
    def has_input(self) -> bool:
        return self._has_input


class IO:
    def get_input_str(self):
        # Construct getopt string
        with_input_str = ''
        without_input_str = 'h'

        for key in self.values:
            value = self.values.get(key)
            if value.has_input:
                with_input_str += key + ':'
            else:
                without_input_str += key

        return with_input_str + without_input_str

    @staticmethod
    def setup_getopt(argv: sys.argv, input_str: str) -> Dict[str, str]:
        try:
            opts, args = getopt.getopt(argv, input_str)
        except getopt.GetoptError:
            print('Command line arguments invalid')
            sys.exit(2)
        return opts

    @staticmethod
    def align_help_strings(commands: List[str], descriptions: List[str]) -> List[str]:
        max_len = 0
        for command in commands:
            if len(command) > max_len:
                max_len = len(command)

        # Add padding
        max_len += 4

        lines = ['', 'Options:']
        for i in range(len(commands)):
            lines.append(
                commands[i] + (' ' * (max_len - len(commands[i]))) + descriptions[i]
            )
        lines.append('')

        return lines

    def show_help(self):
        commands = list()
        descriptions = list()

        for key in self.values:
            command, description = self.values[key].gen_help(key)
            commands.append(command)
            descriptions.append(description)

        for line in IO.align_help_strings(commands, descriptions):
            print(line)

    def loop_args(self, opts: Dict[str, str]):
        for opt, arg in opts:
            if opt == '-h':
                self.show_help()
                self._run = False
                sys.exit(0)
            else:
                found_value = False
                for key in self.values:
                    if opt == '-' + key:
                        self.values[key].update(arg)
                        found_value = True
                        break
                if not found_value:
                    print('Unrecognized argument passed, refer to -h for help')
                    sys.exit(2)

    def get_io(self, argv: sys.argv):
        input_str = self.get_input_str()
        opts = self.setup_getopt(argv, input_str)
        self.loop_args(opts)

    def gen_io_val(self) -> Dict[str, Any]:
        io_val = dict()
        for key in self.values:
            io_val[self.values[key].alias] = self.values[key].value
        return io_val

    def __init__(self, values=Dict[str, IOInput]):
        self.values = values
        self.get_io(sys.argv[1:])
        self.io_val = self.gen_io_val()
        self._run = True

    def get(self, key: str) -> Any:
        return self.io_val.get(key)

    @property
    def run(self) -> bool:
        return self._run
