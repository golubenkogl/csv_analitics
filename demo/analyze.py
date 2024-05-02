#!/usr/bin/env python3

"""
Draft of the base implementation of CSV-LLM-CSV processing module.
This is currently for demonstation purposes only.
"""
from __future__ import annotations
import sys

import argparse
import copy
import csv
import json

from abc import ABC
from functools import lru_cache
from itertools import islice

def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

from pathlib import Path

from dataclasses import (
    dataclass,
    fields,
    field,
    make_dataclass
)

from typing import (
    Generator,
    Any,
    Callable,
    ClassVar,
    get_type_hints,
    Sequence
)

import numpy as np
from validator_collection import validators  # type: ignore[import-untyped]
from LLMProcessor import process

__version__ = "0.1"

__all__ = ["parse_csv", "unparse_csv", "BaseCSVRecord"]

# Default output line buffer size
DEFAULT_BUFFSZ = 300
DEFAULT_DIALECT = "excel"

HEURISTIC_SAMPLE = 20

# The infinity
NUM_INF = float("inf")


@dataclass
class BaseCSVRecord(ABC):
    """CSV record with fixed set of named fields representing the CSV table
    cell, performs basic validations of the field values. Iterable.
    """

    _fieldspec: ClassVar[dict[str, tuple[str, type, Callable, bool]]] = {}

    def __post_init__(self):
        """Post-processes field values."""

        ncols = 0

        try:

            for val_fld, (_, _, val_func, req) in type(self)._fieldspec.items():
                ncols += 1
                self.__setattr__(val_fld, val_func(getattr(self, val_fld), req))
        except ValueError as ex:

            raise ValueError(f"column #{ncols}: {ex}")


    def __iter__(self) -> Generator[str, None, None]:
        """Yields the values of the fields maintaining their order."""

        for fld in fields(self):
            yield getattr(self, fld.name)

    def asdict(self) -> dict:
        """Returns a shallow `dict` representation of the instance, where
        field names are `dict` keys.
        """

        return dict((f.name, getattr(self, f.name)) for f in fields(self))

    @classmethod
    @lru_cache
    def _accept_str(cls, val: str, required: bool = False) -> str:
        """Returns the value coerced to `str` with leading and trailing space
        removed. Raises `ValueError` if `required` is specified and the
        resulting value is an empty string.
        """

        val = str(val).strip()

        if not val and required:
            raise ValueError("non-empty value expected")

        return val

    @classmethod
    @lru_cache
    def _accept_int(cls, val: str, required: bool = False) -> int:
        """Returns the value coerced to `int`. Raises `ValueError` if
        `required` is specified and the specified value is an empty string or
        can't be parsed into `int`.
        """

        val = cls._accept_str(val, required)

        return validators.integer(val)

    @classmethod
    @lru_cache
    def _accept_float(cls, val: str, required: bool = False) -> float:
        """Returns the value coerced to `float`. Raises `ValueError` if
        `required` is specified and the specified value is an empty string or
        can't be parsed into `float`.
        """

        val = cls._accept_str(val, required)

        return validators.float(val)

    @classmethod
    @lru_cache
    def _accept_url(cls, val: str, required: bool = False) -> str:
        """Returns the value if it is a valid URL. Raises `ValueError` if
        `required` is specified and the specified value is an empty string
        or can't be parsed into a URL.
        """

        val = cls._accept_str(val, required)

        return validators.url(val)

    @classmethod
    @lru_cache
    def _accept_email(cls, val: str, required: bool = False) -> str:
        """Returns the value if it is a valid email address. Raises
        `ValueError` if `required` is specified and the specified value is an
        empty string or can't be parsed into an email address.
        """

        val = cls._accept_str(val, required)

        return validators.email(val)

    @classmethod
    @lru_cache
    def _accept_date(cls, val: str, required: bool = False) -> str:
        """Returns the value if it is a structurally valid date. Raises
        `ValueError` if `required` is specified and the specified value is an
        empty string or can't be parsed into a date.
        """

        val = cls._accept_str(val, required)

        return validators.date(val)

    @classmethod
    @lru_cache
    def _accept_datetime(cls, val: str, required: bool = False) -> str:
        """Returns the value if it is a date in ISO 8601 format. Raises
        `ValueError` if `required`
        is specified and the specified value is an empty string or can't be
        parsed into a date in ISO 8601 format.
        """

        val = cls._accept_str(val, required)

        return validators.datetime(val)

    @staticmethod
    def _parse_colspec(*colspec: str) -> tuple[tuple[str, str, str, bool], ...]:
        """Parse the comlum specification in the form "name:[?]T", where
        `name` is the name of the field, `T` is a field value data type, and
        `?` if present signifies that the field can be empty. Returns a tuple
        of, field name (str), original column specification (str), field type
        (str), and whether the field can be empty or not (bool)
        """

        # Using dict to eliminate duplicate column names
        ret: dict[str, tuple[str, str, bool]] = {}

        for fld_num, spec in enumerate(colspec):

            spec_split = spec.split(":")

            split_len = len(spec_split)

            if split_len == 1:
                fld_name = f"col_{fld_num}"

                fld_type = spec_split[0]
            elif split_len == 2:

                fld_name, fld_type = spec_split
            else:
                raise ValueError(f"invalid column specification: {spec}")

            req = True

            if fld_type and fld_type[0] == "?":
                fld_type = fld_type[1:]
                req = False

            if not fld_type:
                raise ValueError(f"no data type in specification: {spec}")

            ret[fld_name] = (spec, fld_type, req)

        return tuple(
            (_name, _spec, _type, _req)
            for _name, (_spec, _type, _req) in ret.items()
        )

    @classmethod
    def derive_class(cls, name: str, *colspec: str) -> type[BaseCSVRecord]:
        """CSV record class factory. Creates CSV record data classes inherited
        from `BaseCSVRecord` and defined according to CSV table column
        specification defined as a list of strings of the format name:[?]T,
        where `name` is the name of the class field (CSV table column), `T`
        is the data type, and `?`, if present, signifies that the value of
        corresponding field can be an empty string
        """

        fieldspec = copy.deepcopy(cls._fieldspec)

        for fld_name, spec, fld_type, req in cls._parse_colspec(*colspec):

            val_func = getattr(cls, f"_accept_{fld_type}")

            func_type = get_type_hints(val_func)["return"]

            fieldspec[fld_name] = (spec, func_type, val_func, req)

        flds: list[tuple[str, type, Any]] = []

        for fld_name, (_, func_type, _, _) in fieldspec.items():
            flds.append((fld_name, func_type, field(default="")))

        # `ClassVar` has type typeing._SpecialForm, represented as <typing
        # special form>, which wierldy enough the type checker sees as an
        # `object`, mypy docs say that is's ok and you don't even need to
        # annotate with ClassVar, but we're not using it yo annotate a
        # class field in our class definition, we're passing it to
        # dataclass.make_dataclass(), and that, type checker is seeing as a
        # type mismatch
        # https://mypy.readthedocs.io/en/stable/class_basics.html

        flds.append(("_fieldspec",
                     ClassVar[dict[str, tuple[str, type, Callable, bool]]],
                     fieldspec))  # type: ignore[arg-type]

        return make_dataclass(name, flds, bases=(cls,), match_args=True)


def parse_csv(
    file: str | Path,
    ranges: Sequence[tuple[int | float, int | float]],
    rec_cls: type[BaseCSVRecord],
    dialect: str | csv.Dialect,
    skiprows: int = 0) -> Generator[BaseCSVRecord, None, None]:
    """Parses a CSV file, the file is expected to be in the specified dialect.
    Yields instances of `rec_cls` class.
    """

    # Unzip ranges, turning e.g. [(1, 2, 3), (4, 5, 6)] sequence into
    # [(1, 4) (2, 5) (3, 6) ]
    # numpy array ops work many times faster than any concievable native
    # Python approaches.
    range_lo, range_hi = [np.array(limit) for limit in zip(*ranges)]

    # Total number of rows
    nrow = 0

    # Number of skipped rows
    nskip = 0

    # Number of parsed rows
    nparse = 0

    with open(file, encoding="utf-8", newline="") as csv_handle:

        # Maximum number of fields in an input record
        nfields = len(rec_cls.__match_args__)

        csv_reader = csv.reader(csv_handle, dialect)

        for row in islice(csv_reader, skiprows, None):
            nrow += 1

            if np.any((range_lo <= nrow) & (nrow <= range_hi)):

                nparse += 1

                try:
                    yield rec_cls(*row[:nfields])
                except (TypeError, ValueError) as ex:
                    nskip += 1
                    # TODO: Proper logging should be used here
                    print(
                        f"Skipping row #{nrow}, reason: {ex}", file=sys.stderr
                    )

    print(
        f"{nskip} of {nparse} parsed of total {nrow} rows were skipped.",
        file=sys.stderr,
    )


def unparse_csv(file: str | Path,
                *records: BaseCSVRecord,
                buffsz: int = 1,
                mode: str = "a"):
    """Unparses one or more `BaseCSVRecord` subclass instancses and writes
    resulting CSV strings to specified file.
    """

    with open(file, mode, encoding="utf-8") as csv_handle:

        csv_writer = csv.writer(csv_handle)

        for buff in batched(records, buffsz):
            csv_writer.writerows(buff)


def parse_ranges(*s_ranges: str) -> Sequence[tuple[int | float, int | float]]:
    """Parses range strings, e.g., 0-5, 20-30, 0-+, into two-tuples of ints,
    translates '+' to `inf` (infinity)."""

    ret = []

    for s_range in s_ranges:
        _range = s_range.split("-", maxsplit=2)

        s_lo = _range[0]

        len_split = len(_range)

        try:

            if len_split == 2:
                s_hi = _range[1]
            elif len_split == 1:
                s_hi = s_lo
            else:
                raise ValueError("invalid range format")

            r_lo = int(s_lo)

            # '$' (dollar sign) would fit this better, it can be interpre-
            # ted by the shell when specified on the cammand line and we
            # don't want that to happen.
            if s_hi == "+":
                r_hi = NUM_INF
            else:
                r_hi = int(s_hi)

            if r_hi < r_lo:
                raise ValueError(
                    (
                        "end endpoint of the range must be greater "
                        "than start endpoint"
                    )
                )

        except (ValueError, TypeError) as ex:
            raise ValueError(f"invalid range: <{s_range}>") from ex

        ret.append((r_lo, r_hi))

    return ret


def read_config(*paths: str | Path) -> dict[str, Any]:
    """Reads JSON configuration files at specified paths in order in which
    they are specified, merges their contents, so that the valus in the every
    next file will override the valus read from the previous one. Returns
    resulting `dict`."""

    ret: dict[str, Any] = {}

    for cfg_path in paths:

        try:
            with open(cfg_path, encoding="utf-8") as cfg_handle:
                ret |= json.load(cfg_handle)

        except (OSError, json.decoder.JSONDecodeError) as ex:
            print(f"cannot load configuration file: '{cfg_path}': {ex}")

    return ret


def get_dialect(fpath: str | Path) -> type[csv.Dialect] | str | None:
    """Tries to guess the CSV file's dialect."""

    # BUG: There's a lot of reports all over the internet, including from the
    # `csv.Sniffer()` original author, saying that `csv.Sniffer().sniff()`
    # can and will return wrong dialects and `csv.Sniffer().has_header()` -
    # false positives and false negatives.

    ret: type[csv.Dialect] | str = DEFAULT_DIALECT

    with open(fpath, encoding="utf-8", newline="") as csv_handle:
        sniffer = csv.Sniffer()

        # Guess the dialect if none specified
        ret = sniffer.sniff(csv_handle.readline())

    return ret


def get_colspec(fpath: str | Path, dialect: csv.Dialect | str) -> list[str]:
    """Tries to guess the CSV file's column specification. Basically reads
    `HEURISTIC_SAMPLE` number of records (sic!, not lines) from a CSV file
    and assume the largest number of culums to be suchc for all table. Returns
    a tuple of strings 'col_1:?str', 'col_2:?str', ..., "col_N:?str".
    """

    # Gess the column specification if none specified
    with open(fpath, encoding="utf-8", newline="") as csv_handle:

        ncols = 0

        csv_reader = csv.reader(csv_handle, dialect)

        for row in islice(csv_reader, 0, HEURISTIC_SAMPLE):
            ncols = max(ncols, len(row))

    return [f"col_{ncol}:?str" for ncol in range(ncols)]


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog="CSV file analyzer",
        description=(
            "Reads records from a CSV file, feed them to an "
            "LLM, produces a CSV file that has both input data "
            "and processing result combined."
        ),
    )

    parser.add_argument(
        "-i",
        "--input",
        help="input CSV file path.",
        type=Path,
        dest="input",
        action="store",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="output CSV file path.",
        type=Path,
        dest="output",
        action="store",
    )

    parser.add_argument(
        "-b",
        "--buffer-size",
        help="output line buffer size.",
        type=int,
        dest="buffsz",
        action="store",
        default=None,
    )

    parser.add_argument(
        "-d",
        "--dialect",
        help="input CSV file dialect.",
        dest="dialect",
        action="store",
        choices=csv.list_dialects(),
        default=None,
    )

    # Command line parameter for range is -n, not -r for semantic reasons, -r
    # usually, especially in POSIX-compliant utilities, stands for recursion
    parser.add_argument(
        "-n",
        "--ranges",
        help=(
            "one or more inclusive ranges of input CSV "
            "file rows that are to be processed, + (plus "
            "sign) signifies the last row, e.g., if -n 0-3 "
            "6 8-+ is specified, the rows 0 through 3, row "
            "6 and rows 8 up to the last row will be "
            "processed."
        ),
        nargs="+",
        dest="ranges",
        default=None,
    )

    parser.add_argument(
        "-c",
        "--column-spec",
        help="column spec",
        nargs="+",
        dest="colspec",
    )

    parser.add_argument(
        "--config",
        help="JSON configuration file path.",
        type=Path,
        action="store",
        dest="config",
    )

    parser.add_argument(
        "-k",
        "--api-key",
        help="LLM API key.",
        action="store",
        dest="apikey",
    )

    parser.add_argument(
        "--dump-config",
        help="print final configuration as JSON",
        action="store_true",
        dest="dumpcfg",
    )

    parser.add_argument(
        "--throw",
        help="raise (throw) exceptions instead of printing fancy errors.",
        action="store_true",
        dest="throw",
    )

    args = parser.parse_args()

    # "Global" configuration file path
    gcfg_path = Path(__file__).resolve().with_name("global.json")

    # If no output file is specified, use the input file name stem
    if args.output is None:
        args.output = args.input.with_stem(f"{args.input.stem}_output")

    # "Local" per-table configuration file path
    if args.config is None:

        # Use table name with .json extension if path is not specified
        lcfg_path = args.input.resolve().with_suffix(".json")
    else:
        lcfg_path = args.config

    try:

        # Read global, then local configuration files
        config = read_config(gcfg_path, lcfg_path)

        print(json.dumps(config, indent=4, default=str))

        # Merge (override) the configuration with specified command line
        # option names and arguments
        config |= dict((k, v) for k, v in vars(args).items() if v is not None)

        # Ingore input and output file paths from configuration
        if "input" in config:
            del config["input"]

        if "output" in config:
            del config["output"]

        # API key is required to be present either in config file(s) or
        # specified as command line argument
        print(json.dumps(config, indent=4, default=str))
        if config.get("apikey") is None:
            raise KeyError("No API key provided.")

        # Prompts amust be speicifed in in config file(s)
        if config.get("prompts") is None:
            raise KeyError("No prompt templates provided.")

        # If no ranges are specified, default to 0-+ (whole file)
        if config.get("ranges") is None:
            config["ranges"] = ["0-+"]

        # If no dialect is specified, try to determine it heurictically from
        # the input file.
        if config.get("dialect") is None:
            config["dialect"] = get_dialect(args.input)

        # If not column specification is specified, make one
        if config.get("colspec") is None:
            config["colspec"] = get_colspec(args.input, config["dialect"])

        # If buffer size is not specified, or not an integer, use default
        # buffer size

        buffsz = DEFAULT_BUFFSZ

        s_buffsz: str | None = config.get("buffsz")

        if s_buffsz is not None:
            try:
                buffsz = max(1, int(s_buffsz))
            except (TypeError, ValueError):
                pass

        config["buffsz"] = buffsz

        # We do not need input and outpiut file paths, dumpcfg, and trow para
        # meters in dumped configuration
        if args.dumpcfg:
            del config["dumpcfg"]
            del config["throw"]

            # Dialect can be an instance of csv.Dialect class, we wnat the
            # dumped configuration to be reusable, so delete it if it is not
            # a 'str'
            if not isinstance(config["dialect"], str):
                del config["dialect"]

            print(json.dumps(config, indent=4, default=str))
        else:

            # Input record class
            rec_cls = BaseCSVRecord.derive_class(
                "InputRecord", *config["colspec"]
            )

            orec_cls = None

            # Parse ranges
            ranges = parse_ranges(*config["ranges"])

            # DEMO:
            csv_parser = parse_csv(
                args.input, ranges, rec_cls, config["dialect"]
            )

            # This here is parses on row at a time, creates a new record with
            # 'result' column, sets its value to the result of processing
            # and immediately appends the record to the output CSV file
            flag = True
            for rec in csv_parser:              
                results = process(rec, config)
                if flag:
                    output_cols = []
                    for i in range(len(results)):
                        output_cols.append("result{i}:?str".format(i=i))
                    orec_cls = rec_cls.derive_class("OutputRecord", *output_cols)
                    flag = False
                orec = orec_cls(*rec, *results)  # type: ignore[call-arg]

                unparse_csv(args.output, orec)

    except (ValueError, TypeError, OSError, KeyError) as ex:
        if args.throw:
            raise

        print(f"error: {ex}", file=sys.stderr, end="\n\n")

        parser.print_help()
