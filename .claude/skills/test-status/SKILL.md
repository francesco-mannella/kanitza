---
description: Check progress of scripts/tests.py running on a simulations folder (default test/base)
disable-model-invocation: true
allowed-tools: Bash
arguments: [root]
---

Report the status of `scripts/tests.py` (which runs `src/test.py` for 2 shapes x 9
rotations = 18 runs per experiment dir, each writing a `goals-*.npy` file on
completion) against `$root/simulations`.

If `$root` is not given, default to `test/base` relative to the repo root.

Steps:
1. Check for a running `tests.py` or `test.py` process: `pgrep -af "scripts/tests.py|src/test.py"`.
2. Resolve `<root>/simulations` and list its experiment subdirectories.
3. For each experiment dir, count `goals-*.npy` files (18 = fully tested, 0 = not
   started, 1-17 = in progress).
4. Summarize: how many dirs are fully tested / in progress / not started (out of
   the total), and if a run is active, which dir it's currently on and its
   file count out of 18.

Keep the report terse — counts and the current in-progress dir, not a per-dir
listing unless asked.
