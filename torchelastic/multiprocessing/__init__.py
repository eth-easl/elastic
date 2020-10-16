#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Multiprocessing library based on subprocess.Popen.
"""

from .api import (  # noqa F401
    Params,
    ProcContext,
    ProcessGroupException,
    TerminationBehavior,
    run,
    run_async,
)
