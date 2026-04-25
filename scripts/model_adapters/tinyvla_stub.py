#!/usr/bin/env python3
from __future__ import annotations

import json
import sys


def main() -> int:
    _ = json.loads(sys.stdin.read() or "{}")
    print(
        "TinyVLA requires a task-specific processed checkpoint and repo-local eval wrapper. "
        "Set VLA_SAFETY_TINYVLA_COMMAND to the concrete inference command for your checkpoint.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
