from typing import Dict, Union, Callable, Any, Generic, TypeVar
import sys
import getopt


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

    def gen_help(self, key: str) -> str:
        help_str = ' -' + key
        if self._has_input:
            help_str += ' <' + self.arg_name + '>'
        else:
            help_str += '       '
        help_str += '   ' + self.descr
        return help_str

    @property
    def has_input(self) -> bool:
        return self._has_input


class IO:
    def get_input_str(self):
        # Construct getopt string
        with_input_str = ''
        without_input_str = ''

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

    def loop_args(self, opts: Dict[str, str]):
        for opt, arg in opts:
            if opt == '-h':
                print('Options:')
                for key in self.values:
                    print(self.values[key].gen_help(key))
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

    def get(self, key: str) -> Any:
        return self.io_val.get(key)
