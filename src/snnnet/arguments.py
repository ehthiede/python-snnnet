"""Utilities for implementing argument / hyperparameter management."""

import argparse
import dataclasses
import json
import typing

from . import utils


@dataclasses.dataclass
class Argument:
    name: str
    type: object
    help: str
    choices: list
    nargs: str
    default: typing.Any


class ArgumentSet:
    """This class encapsulates an argument
    """
    def __init__(self, description=None):
        self.arguments: typing.Dict[str, Argument] = {}
        self.description = description

    def add_argument(self, name, default=None, type=None, help=None, choices=None, nargs=None):
        self.arguments[name] = Argument(name, type, help, choices, nargs, default)

    def get_parser(self, with_default=True) -> argparse.ArgumentParser:
        """Creates an `ArgumentParser` which parses the current set of arguments.

        Parameters
        ----------
        with_default : bool, optional
            If True, indicates that the default values should be specified in the created
            `ArgumentParser`. Otherwise, the created `ArgumentParser` does not specify
            default argument values.

        Returns
        -------
        argparse.ArgumentParser
            An argument parser which can be used to parse arguments as describe by the current `ArgumentSet`.
        """
        parser = argparse.ArgumentParser(description=self.description)

        for arg_name, arg_desc in self.arguments.items():
            option_str = '--' + arg_name

            if arg_desc.type is bool:
                type = utils.str2bool
            else:
                type = arg_desc.type

            # We do not include default in the argument specification
            # so that we can detect arguments that are not specified on the command line
            parser.add_argument(
                option_str,
                type=type,
                help=arg_desc.help,
                choices=arg_desc.choices,
                nargs=arg_desc.nargs,
                default=arg_desc.default if with_default else argparse.SUPPRESS)

        parser.add_argument(
            '--config_file', type=str, default=None,
            help='File from which to load the configuration, specified in JSON format. '
                 'Options specified on the command line override those specified in the file.')

        return parser


    def parse_args(self, args=None):
        """Parse the given arguments, or the command line arguments, according to
        the current `ArgumentSet`.

        Parameters
        ----------
        args: List[str], optional
            If not None, a list of parameters to parser. Otherwise, the parameters
            are read from `sys.argv`.

        Returns
        -------
        dict
            A dictionary mapping argument names to argument values.
        """

        # We will handle defaults ourselves so suppress them in parser
        parser = self.get_parser(with_default=False)
        args = parser.parse_args(args)
        args = vars(args)

        config_file_path = args.get('config_file')
        if config_file_path is not None:
            # if a config file is specified, fill arguments from file
            # in the case that they were not specified at command line.
            with open(config_file_path) as fp:
                config = json.load(fp)

            for k, v in config.items():
                if k not in args:
                    args[k] = self.arguments[k].type(v)

        # fill remaining arguments not specified in file or at command line
        self.fill_dictionary_with_defaults(args)

        return args

    def fill_dictionary_with_defaults(self, config: dict):
        """Fills the given dictionary with the default values
        of the current `ArgumentSet` where the values are not specified.

        Parameters
        ----------
        config : dict
            A dictionary containing the configuration

        Returns
        -------
        dict
            The same `config` dictionary, modified to have all defaults be present.
        """
        for arg_name, arg_desc in self.arguments.items():
            config.setdefault(arg_name, arg_desc.default)
        return config
