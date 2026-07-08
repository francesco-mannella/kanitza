---
description: Send a status update for the active kanitza run (or a custom message) to laptop's desktop via ssh + remote xmsg
disable-model-invocation: true
allowed-tools: Bash
arguments: [message]
---

Send a message to the remote host `laptop`'s active graphical session via its
`xmsg` popup tool:

```
MESSAGE="..."
nohup ssh laptop "DISPLAY=:0 python3 /home/fmannella/bin/xmsg -t \"$MESSAGE\"" > /tmp/xmsg_last.log 2>&1 &
disown
```

**Always dispatch it in the background exactly like that — `nohup` plus `&`
plus `disown`, all three.** `xmsg` blocks until the popup is manually
dismissed on the remote desktop — running it in the foreground will hang the
calling command indefinitely if nobody's there to close it (confirmed:
popups can stack up unacknowledged). `nohup` stops a SIGHUP from killing the
ssh client if the local shell exits/disconnects before the popup is
dismissed; `disown` keeps it out of the shell's job table. Never wait on it
synchronously, especially inside a loop. After dispatching, you can sanity
check it didn't die immediately with `pgrep -af "xmsg"` on laptop — it should
still be listed (waiting for dismissal) rather than absent.

`xmsg` needs a live X session to pop up in — check first with
`ssh laptop who` (look for a `(:0)` session) if this fails.

**Backtick gotcha (confirmed to actually break silently):** if `$MESSAGE`
contains markdown code-span backticks (`` `like this` ``), they MUST be
escaped as `\`` when building the `MESSAGE=` assignment, e.g.
`MESSAGE="... \`sweep_d3to4\` ..."`. Unescaped backticks inside the
double-quoted `MESSAGE=` string trigger bash command substitution — it
tries to execute the backticked text as a shell command, that command
doesn't exist, and `xmsg` silently never launches (no popup, no error
surfaced locally since the failure happens in the backgrounded/detached
remote shell). After dispatching, always verify with
`ssh laptop 'pgrep -af "bin/xmsg"'` that the process is actually running
with the intended text — don't just trust that the local `ssh ... &`
returned without an error.

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
   is in progress. Also estimate an ETA: total target combos = 18 x number
   of experiment dirs; completed combos = sum of goals-*.npy counts across
   all dirs; elapsed time from the process's `ps -o etime= -p <pid>`; rate =
   completed / elapsed seconds; remaining = (total - completed) / rate.
4. If neither process is running, report that no kanitza run is currently
   active.
5. Compose the message as clear, well-understandable prose using inline
   markdown for emphasis — not terse symbol-only notation. Spell out
   parameter names in full (`decaying_speed=3.0`, not `d:3.0`), use
   **bold** for the key numbers (progress, %, ETA) and backtick `code
   spans` for identifiers (root/folder names, parameter names), and write
   it as a real sentence or two rather than a compressed fragment. Mention
   the root/params if identifiable from the experiment directory names or
   `final_parameters`, and the progress. **Always include a predicted end
   date/time** (absolute, e.g. "ETA **~15:38 CEST**" or "**~2026-07-07 08:40
   CEST**" if it crosses midnight), computed from current time + the
   remaining-time estimate above — not just a relative "X% done". If no run
   is active, skip the ETA.

Example style (xmsg displays raw text, so the markdown characters show
literally — that's fine, they still make the structure and emphasis easy to
parse at a glance):

"**Kanitza** training on `sweep_d3to4` (decaying_speed=3.0, batch 1/7):
**143/500** epochs (~29%), overall **~5%** done. ETA **2026-07-08 09:42
CEST**."

Prioritize clarity over brevity — a two-sentence, fully-spelled-out message
is better than a cryptic one-liner, but don't pad it with irrelevant detail.
