#!/usr/bin/env python3
"""Compatibility entry point for explicitly labeled locked-policy replay.

The previous script fit three logistic heads on the same trajectories it then
evaluated. It did not call a generator or measure GPU savings/decision latency.
Those in-sample replay results must not be described as live stopping.

This entry point now delegates to the prefix-only evaluation tooling. With no
arguments it runs saved-trajectory replay, with measured completion and second-
path lengths and gold confined to evaluation. For an actual local model run use
``python research/run_online_stopping_evaluation.py --live``; the new collector
freezes policy/tasks before generation and stops scheduling future steps.
"""

from __future__ import annotations

import sys

from run_online_stopping_evaluation import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("--replay")
    main()
