---
description: Send a status update for the active kanitza run (or a custom message) to mind1's desktop via ssh + remote xmsg
disable-model-invocation: true
allowed-tools: Bash
arguments: [message]
---

Send a one-line message to the remote host `mind1`'s active graphical session
via its `xmsg` popup tool:

```
MESSAGE="..."
ssh mind1 "DISPLAY=:0 python3 /home/fmannella/bin/xmsg -t \"$MESSAGE\"" > /tmp/xmsg_last.log 2>&1 &
disown
```

**Always dispatch it in the background exactly like that.** `xmsg` blocks
until the popup is manually dismissed on the remote desktop — running it in
the foreground will hang the calling command indefinitely if nobody's there
to close it (confirmed: popups can stack up unacknowledged). Never wait on it
synchronously, especially inside a loop.

`xmsg` needs a live X session to pop up in — check first with
`ssh mind1 who` (look for a `(:0)` session) if this fails.

If `$message` is given, use it verbatim as `$MESSAGE` and send it directly —
skip the detection steps below.

Otherwise, auto-detect the current kanitza status:

1. Check for a running training or evaluation process:
   `pgrep -af "grid_search.py|scripts/tests.py"`.
2. If a `grid_search.py` (training) process is running: find its root (the
   `test/<name>` directory it was launched from — check the process's cwd via
   `ls -l /proc/<pid>/cwd`, or infer from which `simulations/*/log` files are
   being actively written). For each experiment dir under
   `<root>/simulations/`, count completed epochs with
   `grep -c "^epoch: " <dir>/log`. Compute the average epoch count, the %
   of the target (`epochs` value in that dir's `final_parameters` or
   `loaded_params`, commonly 500), and estimate an ETA from elapsed wall-clock
   time (`ps -o etime= -p <pid>`) versus epochs completed (rate = epochs /
   elapsed seconds; remaining = (target - avg_epochs) * seconds/epoch).
3. If a `scripts/tests.py` (evaluation) process is running instead: count
   `goals-*.npy` files per experiment dir under its root's `simulations/`
   (18 = fully tested per dir) and report how many dirs are fully tested /
   in progress / not started, plus the current file count of whichever dir
   is in progress.
4. If neither process is running, report that no kanitza run is currently
   active.
5. Compose one concise line — mention the root/params if identifiable from
   the experiment directory names or `final_parameters` (e.g. `m:`/`d:`/`l:`
   for `match_std`/`decaying_speed`/`local_decaying_speed`), the progress,
   and an ETA if computed — and send it as `$MESSAGE`.

Keep the message terse: a single line, under ~150 characters, in the style
used in this session, e.g.:
"kanitza seed sweep (m:8.0 d:3.0 l:1.5, seeds 101-104): 465/500 epochs avg
(~93%), ETA ~15:38 CEST"
