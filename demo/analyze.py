#!/usr/bin/env python3

"""
Draft of the base implementation of CSV-LLM-CSV processing module.
This is currently for demonstation purposes only.
"""

from dataclasses import dataclass, fields, field, Field
from pathlib import Path
from typing import Generator, Optional
from urllib.parse import urlparse

import sys
import argparse
import csv
import numpy as np

__version__ = "0.1"

__all__ = ["CSVRecord", "process_csv"]

# Default output line buffer size
DEFAULT_BUFFSZ = 300

# The infinity
NUM_INF = float("inf")


@dataclass(order=True)
class CSVRecord:
    """CSV record with fixed set of named fields representing the CSV table
    cell, performs basic validations of the field values. Iterable.
    """

    # Fields
    hit: str
    exhib: str
    country: str
    zip_code: str
    city: str
    state: str
    district: str
    company_url: str
    booth: str
    exhib_url: str
    unk_a: Optional[str] = field(default=None)
    unk_b: Optional[str] = field(default=None)
    comment: Optional[str] = field(default=None)
    unk_c: Optional[str] = field(default=None)
    result: Optional[str] = field(default=None)


    def __post_init__(self):
        """Post-processes field values."""

        if not self.exhib:
            raise ValueError("Missing 'Exhibitor' field")

        for url in (self.company_url, self.exhib_url):
            parse_result = urlparse(url)

            if not parse_result.scheme or not parse_result.netloc:
                raise ValueError(f"Invalid URL for <{self.exhib}>: <{url}>")

    def fields(self) -> Generator[Field, None, None]:
        """Returns the fields of the instance."""

        yield from fields(self)


    def asdict(self) -> dict:
        """Returns a shallow `dict` representation of the instance, where
        field names are `dict` keys.
        """

        return dict((f.name, getattr(self, f.name)) for f in self.fields())

    def interpolate(self, template: str) -> str:
        """Interpolates the template string that may contain field names as
        format placeholders.
        """

        return template.format(**self.asdict())

    def __iter__(self) -> Generator[str, None, None]:
        """Yields the values of the fields maintaining their order."""

        for fld in fields(self):
            yield getattr(self, fld.name)


def parse_csv(
    file: str | Path,
    ranges: list[tuple[int | float, int | float]],
    dialect: str | csv.Dialect = "excel") -> Generator[CSVRecord,
                                                       None,
                                                       None]:
    """Parses a CSV file, the file is expected to be in the specified dialect.
    Default dialect is 'excel'. Yields instances of CSVRecord class.
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
        csv_reader = csv.reader(csv_handle, dialect=dialect)

        for row in csv_reader:
            nrow += 1

            if np.any((range_lo <= nrow) & (nrow <= range_hi)):

                nparse += 1

                try:
                    yield CSVRecord(*row[:13])
                except (TypeError, ValueError) as ex:
                    nskip += 1
                    # TODO: Proper logging should be used here
                    print(f"Skipping row #{nrow}, reason: {ex}",
                          file=sys.stderr)

    print(f"{nskip} of {nparse} parsed of total {nrow} rows were skipped.",
          file=sys.stderr)


def unparse_csv(file: str | Path, *records: CSVRecord):
    """Unparses one or more CSVRecord class instancses and writes resulting
    CSV strings to specified file.
    """

    with open(file, "w", encoding="utf-8") as csv_handle:
        csv_writer = csv.writer(csv_handle)
        csv_writer.writerows(records)


def process_csv(
    ifile: str | Path,
    ofile: str | Path,
    ranges: list[tuple[int | float, int | float]],
    dialect: str | csv.Dialect = "excel",
    buffsz: int = -1):

    """Processes an input CSV files, does required magic, then writes the
    resulting CSV file. The input file is expected to be, and the output
    file will be written in the specified dialect. Default dialect is
    'excel'. Uses beffer of size `buffsz`, for optimal performance keep
    that to at least 10% of the expected total number of lines in the
    output file.
    """

    buff = []

    if buffsz < 0:
        buffsz = DEFAULT_BUFFSZ

    for rec in parse_csv(ifile, ranges=ranges, dialect=dialect):

        # MOCKUP:
        # That's your formatted prompt and the result of processing here
        # You can call your LLM API here or pass a calling function or class
        # as argument (not implemented).
        rec.result = rec.interpolate(
            ("Processed {exhib} from booth " "{booth} with URL {exhib_url}")
        )

        buff.append(rec)

        if len(buff) >= buffsz:
            unparse_csv(ofile, *buff)

            # Clear the buffer
            del buff[:]

    # Flush whatever's left in the buffer
    if buff:
        unparse_csv(ofile, *buff)


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
        required=True,
    )

    parser.add_argument(
        "-b",
        "--buffer-size",
        help="output line buffer size.",
        type=int,
        dest="buffsz",
        action="store",
        default=DEFAULT_BUFFSZ,
    )

    # Command line option for range is -n, not -r for semantic reasons, -r
    # usually, especially in POSIX-compliant utilities, stands for recursion
    parser.add_argument(
        "-n",
        "--range",
        help=(
            "one or more inclusive ranges of input CSV "
            "file rows that are to be processed, + (plus "
            "sign) signifies the last row, e.g., if -n 0-3 "
            "6 8-+ is specified, the rows 0 through 3, row "
            "6 and rows 8 up to the last row will be "
            "processed."
        ),
        nargs="+",
        default=["0-+"],
    )

    parser.add_argument(
        "--throw",
        help="raise (throw) exceptions instead of printing fancy errors.",
        action="store_true",
        dest="throw",
    )

    args = parser.parse_args()

    # Normalized ranges
    norm_ranges: list[tuple[int | float, int | float]] = []

    try:

        # Convert and normalize ranges
        for s_range in args.range:
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

            norm_ranges.append((r_lo, r_hi))


            # Time for magic.
            process_csv(args.input,
                        args.output,
                        norm_ranges,
                        buffsz=args.buffsz)

    except (ValueError, TypeError, OSError) as ex:
        if args.throw:
            raise

        print(f"error: {ex}", file=sys.stderr, end="\n\n")

        parser.print_help()
