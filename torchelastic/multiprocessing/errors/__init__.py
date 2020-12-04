#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import signal
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from string import Template
from typing import Dict, List, Optional, Tuple

from torchelastic.multiprocessing.errors.error_handler import ErrorHandler  # noqa F401
from torchelastic.multiprocessing.errors.handlers import get_error_handler  # noqa F401


JSON = Dict

_EMPTY_ERROR_DATA = {"message": "<NONE>"}
_NOT_AVAILABLE = "<N/A>"


@dataclass
class ProcessFailure:
    """
    Represents the failed process result. When the worker process fails,
    it may record failure root cause into the file.
    Tries to read the failure timestamp from the provided ``error_file``,
    if the ``error_file`` does not exist, the timestamp is the current
    timestamp (seconds since epoch).

    The ``message`` field is a concise explanation of the failure. If
    the error file exists then the message is obtained from the error file.
    Otherwise one is generated based on the failure signature.

    .. note:: It is assumed that the ``error_file`` is written by
              ``torchelastic.multiprocessing.errors.error_handler.ErrorHandler``.
              Otherwise the behavior is undefined.

    """

    local_rank: int
    pid: int
    exitcode: int
    error_file: str
    error_file_data: JSON = field(init=False)
    message: str = field(init=False)
    timestamp: int = field(init=False)

    def __post_init__(self):
        if os.path.isfile(self.error_file):
            with open(self.error_file, "r") as fp:
                self.error_file_data = json.load(fp)
                self.message = self.error_file_data["message"]["message"]
                self.timestamp = int(
                    self.error_file_data["message"]["extraInfo"]["timestamp"]
                )
        else:
            self.error_file = _NOT_AVAILABLE
            self.error_file_data = _EMPTY_ERROR_DATA
            self.message = ""
            self.timestamp = int(time.time())

        # make up an informative message if not already present
        if not self.message:
            # signals typically do not generate an error file message
            if self.exitcode < 0:
                self.message = (
                    f"Signal {-self.exitcode} ({self.signal_name()})"
                    f" received by PID {self.pid}"
                )
            else:
                self.message = f"Process failed with exitcode {self.exitcode}"

    def signal_name(self) -> str:
        if self.exitcode < 0:
            return signal.Signals(-self.exitcode).name
        else:
            return _NOT_AVAILABLE

    def timestamp_isoformat(self):
        """
        Returns timestamp in ISO format (YYYY-MM-DD_HH:MM:SS)
        """
        return datetime.fromtimestamp(self.timestamp).isoformat(sep="_")


GlobalRank = int

_FAILURE_FORMAT_TEMPLATE = """[${idx}]:
  time: ${time}
  rank: ${rank} (local_rank: ${local_rank})
  exitcode: ${exitcode} (pid: ${pid})
  error_file: ${error_file}
  msg: \"${message}\""""

# extra new lines before and after are intentional
_MSG_FORMAT_TEMPLATE = """
${boarder}
${title}
${section}
Root Cause:
${root_failure}
${section}
Other Failures:
${other_failures}
${boarder}
"""


class ChildFailedError(Exception):
    """
    Special exception type that can be raised from a function annotated with the
    ``@record`` decorator to have the child process' (root exception) propagate
    up the stack as-is (e.g. without being wrapped in the parent's traceback).

    Useful in cases where the parent is a simple nanny process
    and the child (worker) processes are actually doing meaningful compute.
    In this case, errors typically occur on the child process as the parent
    is not doing anything non-trivial, and child errors should be propagated
    to the scheduler for accurate root cause diagnostics.

    .. note:: The propagation relies on error files rather than exception handling to
              support both function and binary launches.

    Example:

    ::

     # process tree on a host (container)
     0: scheduler-init-process:
                |- 1: torchelastic_agent:
                         |- 2: trainer_0 (ok)
                         |- 3: trainer_1 (fail) -> error.json
                         |- ...
                         |- n+2: trainer_n (ok)
                |- n+3: other processes
                |- ...

    In the example above, trainer 1's failure (written into error.json) is
    the root cause and should be reported to the scheduler's init process.
    The torchelastic agent raises a ``ChildFailedError("trainer", {1: "trainer_1/error.json"})``
    upon detecting trainer 1's failure which would propagate the contents
    of trainer 1's error file to the scheduler's init process.
    """

    def __init__(self, name: str, failures: Dict[GlobalRank, ProcessFailure]):
        self.name = name
        self.failures = failures
        assert (
            self.failures
        )  # does not make sense to create a ChildFaileError with no failures
        super().__init__(self.format_msg())

    def get_first_failure(self) -> Tuple[GlobalRank, ProcessFailure]:
        rank = min(self.failures.keys(), key=lambda r: self.failures[r].timestamp)
        return rank, self.failures[rank]

    def format_msg(self, boarder_delim="*", section_delim="="):
        title = f"  {self.name} FAILED  "
        root_rank, root_failure = self.get_first_failure()

        root_failure_fmt: str = ""
        other_failures_fmt: List[str] = []
        width = len(title)
        for idx, (rank, failure) in enumerate(self.failures.items()):
            fmt, w = self._format_failure(idx, rank, failure)
            width = max(width, w)
            if rank == root_rank:
                root_failure_fmt = fmt
            else:
                other_failures_fmt.append(fmt)

        return Template(_MSG_FORMAT_TEMPLATE).substitute(
            boarder=boarder_delim * width,
            title=title.center(width),
            section=section_delim * width,
            root_failure=root_failure_fmt,
            other_failures="\n".join(other_failures_fmt or ["  <NO_OTHER_FAILURES>"]),
        )

    def _format_failure(
        self, idx: int, rank: int, failure: ProcessFailure
    ) -> Tuple[str, int]:
        fmt = Template(_FAILURE_FORMAT_TEMPLATE).substitute(
            idx=idx,
            time=failure.timestamp_isoformat(),
            rank=rank,
            local_rank=failure.local_rank,
            exitcode=failure.exitcode,
            pid=failure.pid,
            error_file=failure.error_file,
            message=failure.message,
        )
        width = 0
        for line in fmt.split("\n"):
            width = max(width, len(line))
        return fmt, width


def _no_error_file_warning_msg(rank: int, failure: ProcessFailure) -> str:
    msg = [
        "CHILD PROCESS FAILED WITH NO ERROR_FILE"
        f"Child process {failure.pid} (local_rank {rank}) FAILED (exitcode {failure.exitcode})"
        f"Error msg: {failure.message}",
        f"Without writing an error file to {failure.error_file}.",
        "While this DOES NOT affect the correctness of your application,",
        "no trace information about the error will be available for inspection.",
        "Consider decorating your top level entrypoint function with",
        "torchelastic.multiprocessing.errors.record. Example:",
        "",
        r"  from torchelastic.multiprocessing.errors import record",
        "",
        r"  @record",
        r"  def trainer_main(args):",
        r"     # do train",
    ]
    width = 0
    for line in msg:
        width = max(width, len(line))

    boarder = "*" * width
    header = "CHILD PROCESS FAILED WITH NO ERROR_FILE".center(width)
    return "\n".join(["\n", boarder, header, boarder, *msg, boarder])


def record(fn, error_handler: Optional[ErrorHandler] = None):
    """
    Syntactic sugar to record errors/exceptions that happened in the decorated
    function using the provided ``error_handler``.

    Using this decorator is equivalent to:

    ::

     error_handler = get_error_handler()
     error_handler.initialize()
     try:
        foobar()
     except Exception as e:
        error_handler.record(e)


    .. important:: use this decorator once per process at the top level method,
                   typically this is the main method.

    Example

    ::

     @record
     def main():
         pass

     if __name__=="__main__":
        main()

    """

    if not error_handler:
        error_handler = get_error_handler()

    def wrap(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            error_handler.initialize()
            try:
                return f(*args, **kwargs)
            except ChildFailedError as e:
                rank, failure = e.get_first_failure()
                if failure.error_file != _NOT_AVAILABLE:
                    error_handler.copy_error_file(failure.error_file)
                else:
                    warnings.warn(_no_error_file_warning_msg(rank, failure))
                raise
            except Exception as e:
                error_handler.record_exception(e)
                raise

        return wrapper

    return wrap(fn)
