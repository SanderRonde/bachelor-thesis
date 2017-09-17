import sys
from typing import Dict, Union, Callable, Any, Generic, TypeVar, Tuple, List

T = TypeVar('T')


class IOInput(Generic[T]):
    def __init__(self, default_value: T, data_type: Union[type, Callable[[str], T]],
                 has_input=True, arg_name=None, descr=None, alias=None, is_generic=False):
        self.default_value = default_value
        self.data_type = data_type
        self.value = default_value

        self._has_input = has_input
        self.arg_name = arg_name
        self.descr = descr
        self.alias = alias
        self.is_generic = is_generic

    def update(self, arg: str):
        if (arg.startswith("'") or arg.startswith('"')) and (arg.endswith("'") or arg.endswith("'")):
            arg = arg[1:-1]
        if self.data_type == str:
            self.value = arg
        elif self.data_type == int:
            self.value = int(arg)
        elif self.data_type == float:
            self.value = float(arg)
        elif self.data_type == bool:
            self.value = not self.value
        elif self.data_type == list:
            if self.value is None:
                self.value = list()
            self.value.append(arg)
        else:
            self.value = self.data_type(arg)

    def gen_help(self, key: str) -> Tuple[str, str]:
        help_str = ' -' + key
        if self._has_input:
            help_str += ' <' + self.arg_name + '>'
        help_str += ',' + ' --' + self.alias
        if self._has_input:
            help_str += ' <' + self.arg_name + '>'
        return help_str, self.descr

    @property
    def has_input(self) -> bool:
        return self._has_input


class IO:
    @staticmethod
    def dash_alias(alias: str):
        return alias.replace('_', '-')

    def find_args(self, argv: sys.argv):
        skip = False
        for i in range(len(argv)):
            if skip:
                skip = False
                continue

            arg = argv[i]
            if arg == '-h':
                self.show_help()
                self._run = False
                sys.exit(0)
            else:
                found = False
                for key in self.values:
                    if arg == '-' + key:
                        if self.values[key].has_input:
                            self.values[key].update(argv[i + 1])
                            skip = True
                        else:
                            self.values[key].update('Toggle')
                            skip = False
                        found = True
                        break
                    if '--'in arg and self.values[key].alias == arg[2:] and '=' not in arg:
                        self.values[key].update('Toggle')
                        found = True
                    if '=' in arg and (arg.split('=', 1)[0] == '--' + self.values[key].alias or
                        arg.split('=', 1)[0] == '--' + self.dash_alias(self.values[key].alias)):
                        self.values[key].update(arg.split('=', 1)[1])
                        found = True
                        break

                if not found:
                    if not arg.startswith('-') and '__generic__' in self.values:
                        self.values['__generic__'].update(arg)
                    elif not arg.startswith('-'):
                        print('Found unexpected argument', arg)
                        print('Generic values are not enabled, refer to -h for help')
                        sys.exit(2)
                    else:
                        print('Unrecognized argument passed, refer to -h for help')
                        sys.exit(2)

    @staticmethod
    def align_help_strings(pre_options: List[str], commands: List[str], descriptions: List[str]) -> List[str]:
        max_len = 0
        for command in commands:
            if len(command) > max_len:
                max_len = len(command)

        # Add padding
        max_len += 4

        lines = pre_options + ['', 'Options:']
        for i in range(len(commands)):
            lines.append(
                commands[i] + (' ' * (max_len - len(commands[i]))) + descriptions[i]
            )
        lines.append('')

        return lines

    def show_help(self):
        commands = list()
        descriptions = list()

        has_generic = False
        for key in self.values:
            if key == '__generic__':
                has_generic = True
                continue
            command, description = self.values[key].gen_help(key)
            commands.append(command)
            descriptions.append(description)

        pre_options = list()
        if has_generic:
            filename = sys.argv[0]
            pre_options.append('Synthax: ' + filename + ' [options] [' +
                               self.values['__generic__'].alias + '...]')

        for line in IO.align_help_strings(pre_options, commands, descriptions):
            print(line)

    def gen_io_val(self) -> Dict[str, Any]:
        io_val = dict()
        for key in self.values:
            io_val[self.values[key].alias] = self.values[key].value
        return io_val

    def __init__(self, values=Dict[str, IOInput]):
        self.values = values
        self.find_args(sys.argv[1:])
        self.io_val = self.gen_io_val()
        self._run = True

    def get(self, key: str) -> Any:
        return self.io_val.get(key)

    def get_all(self) -> Dict[str, any]:
        return self.io_val

    @property
    def run(self) -> bool:
        return self._run
