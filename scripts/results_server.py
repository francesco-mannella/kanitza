"""Local web server that shows all the results of a simulation folder.

Usage:
    python /path/to/scripts/results_server.py [FOLDER] [--port 8000]
        [--host ADDRESS]

Open the printed address (http://<host>:8000) and type (or pick) a
simulation folder in the
form, or press "Browse..." to navigate the folders below the launch
directory in a popup (simulation folders have a Select button). FOLDER, if
given, is shown first; the folder list offers every directory under the
launch directory (two levels deep) containing data_sim, a log or
loaded_params. By default the server listens on this machine's Tailscale
IPv4 (from `tailscale ip -4`), so it is reachable from the tailnet, or on
127.0.0.1 if Tailscale is not available; --host overrides it.

If the folder has a data_sim subfolder (runs logged by src/local_wandb.py,
i.e. main.py/test.py/test_generative.py run without -w/--wandb), the page
shows everything logged there:
    - a summary and the list of runs (training and test);
    - one chart per logged metric (competence, weight changes, ...) plus
      the salient samples per epoch, with training runs merged by step so
      a resumed training reads as one curve;
    - the logged maps snapshots (click for the gif) and other media;
    - for each test run: world, pose, goal sequence (from its goals file)
      and its animations;
    - the configuration of the latest training run.
Otherwise (older runs, or runs logged to wandb) it falls back to log,
loaded_params, maps_*.png/gif, sim_*.gif, goals*.npy and *_test_* gifs.
The last lines of log and nohup.out are shown in both cases.

Only simulation folders are shown, only images directly inside them or
under their data_sim are served.
"""
import argparse
import datetime
import html
import json
import os
import re
import subprocess
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, quote, urlparse

import numpy as np


STYLE = """
:root {
  color-scheme: light;
  --surface: #fcfcfb; --surface-2: #f3f2ef; --border: #dcdbd6;
  --text: #0b0b0b; --text-2: #52514e; --grid: #e6e5e0;
  --series-1: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface: #1a1a19; --surface-2: #242423; --border: #3a3a37;
    --text: #ffffff; --text-2: #c3c2b7; --grid: #2e2e2c;
    --series-1: #3987e5;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --surface-2: #242423; --border: #3a3a37;
  --text: #ffffff; --text-2: #c3c2b7; --grid: #2e2e2c;
  --series-1: #3987e5;
}
body { background: var(--surface); color: var(--text); margin: 0;
  font: 14px/1.45 system-ui, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 16px; }
h1 { font-size: 20px; } h2 { font-size: 16px; margin-top: 28px; }
form { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
input[type=text] { flex: 1; min-width: 240px; padding: 6px 8px;
  background: var(--surface-2); color: var(--text);
  border: 1px solid var(--border); border-radius: 6px; }
button { padding: 6px 14px; border-radius: 6px; border: 1px solid var(--border);
  background: var(--surface-2); color: var(--text); cursor: pointer; }
.muted { color: var(--text-2); }
.tiles { display: flex; gap: 12px; flex-wrap: wrap; }
.tile { border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; }
.tile strong { display: block; font-size: 22px; }
.chart { position: relative; border: 1px solid var(--border); border-radius: 8px;
  padding: 8px; margin-top: 8px; }
.chart svg { width: 100%; height: auto; display: block; touch-action: none; }
.tooltip { position: absolute; pointer-events: none; display: none;
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 6px;
  padding: 4px 8px; font-size: 12px; white-space: nowrap; }
.gallery { display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); }
.gallery figure { margin: 0; }
.gallery img { width: 100%; border: 1px solid var(--border); border-radius: 6px; }
figcaption { font-size: 12px; color: var(--text-2); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
td, th { border-bottom: 1px solid var(--border); padding: 3px 6px; text-align: left;
  vertical-align: top; }
pre { background: var(--surface-2); padding: 8px; border-radius: 6px;
  overflow-x: auto; font-size: 12px; }
details { margin-top: 6px; }
dialog { background: var(--surface); color: var(--text); width: min(640px, 92vw);
  border: 1px solid var(--border); border-radius: 10px; padding: 0; }
dialog::backdrop { background: rgb(0 0 0 / 0.4); }
.dlg-head { padding: 12px 16px; border-bottom: 1px solid var(--border);
  display: flex; gap: 8px; align-items: center; }
.dlg-path { flex: 1; font-family: monospace; font-size: 12px; overflow-wrap: anywhere; }
.dlg-list { max-height: 60vh; overflow-y: auto; margin: 0; padding: 4px 0; list-style: none; }
.dlg-list li { display: flex; align-items: center; gap: 8px; padding: 4px 16px; }
.dlg-list li:hover { background: var(--surface-2); }
.dlg-list a { flex: 1; color: var(--text); text-decoration: none; cursor: pointer; }
.badge { font-size: 11px; color: var(--text-2); border: 1px solid var(--border);
  border-radius: 10px; padding: 0 6px; }
"""

BROWSE_JS = """
var dlg = document.getElementById('browser');
var input = document.querySelector('input[name=folder]');
function choose(path) { input.value = path; dlg.close(); input.form.submit(); }
function button(label, onclick) {
  var b = document.createElement('button');
  b.type = 'button'; b.textContent = label; b.onclick = onclick; return b;
}
function load(path) {
  fetch('/browse?path=' + encodeURIComponent(path || '')).then(function (r) {
    return r.json();
  }).then(function (d) {
    var head = dlg.querySelector('.dlg-head'), list = dlg.querySelector('.dlg-list');
    head.textContent = ''; list.textContent = '';
    if (d.parent !== null) head.appendChild(button('Up', function () { load(d.parent); }));
    var p = document.createElement('span'); p.className = 'dlg-path'; p.textContent = d.path;
    head.appendChild(p);
    if (d.sim) head.appendChild(button('Select this folder', function () { choose(d.path); }));
    head.appendChild(button('Close', function () { dlg.close(); }));
    if (d.error) {
      var e = document.createElement('li'); e.textContent = d.error; list.appendChild(e);
    }
    d.dirs.forEach(function (x) {
      var li = document.createElement('li'), a = document.createElement('a');
      a.textContent = x.name + '/';
      a.title = x.sim ? 'Show this simulation' : 'Open this folder';
      a.onclick = x.sim ? function () { choose(x.path); } : function () { load(x.path); };
      li.appendChild(a);
      if (x.sims) {
        var c = document.createElement('span'); c.className = 'badge';
        c.textContent = x.sims + (x.sims === 1 ? ' simulation' : ' simulations') + ' inside';
        li.appendChild(c);
      }
      if (x.sim) {
        var b = document.createElement('span'); b.className = 'badge';
        b.textContent = 'simulation'; li.appendChild(b);
        li.appendChild(button('Open folder', function () { load(x.path); }));
        li.appendChild(button('Select', function () { choose(x.path); }));
      }
      list.appendChild(li);
    });
    if (!d.dirs.length && !d.error) {
      var li = document.createElement('li'); li.className = 'muted';
      li.textContent = 'No subfolders.'; list.appendChild(li);
    }
  });
}
document.getElementById('browse').onclick = function () {
  dlg.showModal(); load(input.value);
};
"""

CHART_JS = """
document.querySelectorAll('.chart').forEach(function (box) {
  var data = JSON.parse(box.dataset.points);
  var svg = box.querySelector('svg'), tip = box.querySelector('.tooltip');
  var line = svg.querySelector('.cross'), dot = svg.querySelector('.hover-dot');
  var g = JSON.parse(box.dataset.geom);
  function sx(x) { return g.left + (x - g.xmin) / (g.xmax - g.xmin) * g.pw; }
  function sy(y) { return g.top + g.h - (y - g.ymin) / (g.ymax - g.ymin) * g.h; }
  function show(evt) {
    var r = svg.getBoundingClientRect();
    var x = g.xmin + ((evt.clientX - r.left) * g.w / r.width - g.left) / g.pw * (g.xmax - g.xmin);
    var lo = 0, hi = data.length - 1;
    while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (data[mid][0] < x) lo = mid; else hi = mid; }
    var p = Math.abs(data[lo][0] - x) <= Math.abs(data[hi][0] - x) ? data[lo] : data[hi];
    var px = sx(p[0]), py = sy(p[1]);
    line.setAttribute('x1', px); line.setAttribute('x2', px);
    dot.setAttribute('cx', px); dot.setAttribute('cy', py);
    line.style.display = dot.style.display = '';
    tip.textContent = '';
    var b = document.createElement('strong'); b.textContent = p[2];
    tip.appendChild(b);
    tip.appendChild(document.createTextNode('  ' + g.xlabel + ' ' + p[0] + (p[3] ? '  ' + p[3] : '')));
    tip.style.display = 'block';
    var left = px * r.width / g.w + 12;
    if (left + tip.offsetWidth > r.width) left -= tip.offsetWidth + 24;
    tip.style.left = left + 'px'; tip.style.top = '8px';
  }
  svg.addEventListener('pointermove', show);
  svg.addEventListener('pointerleave', function () {
    tip.style.display = 'none'; line.style.display = dot.style.display = 'none';
  });
});
"""


def read_lines(path):
    try:
        with open(path, errors="replace") as f:
            return f.read().splitlines()
    except FileNotFoundError:
        return []


def parse_log(folder):
    """Return per-epoch lists: competence, salient samples, object label."""
    comp, samples, labels = [], [], []
    for line in read_lines(os.path.join(folder, "log")):
        if line.startswith("comp:"):
            comp.append(float(line.split(":", 1)[1]))
        m = re.match(r"triangles: (\d+), squares: (\d+)", line)
        if m:
            t, s = int(m.group(1)), int(m.group(2))
            samples.append(t + s)
            labels.append("triangle" if t >= s else "square")
    return comp, samples, labels


def epoch_of(name):
    m = re.search(r"_(\d+)\.", name)
    return int(m.group(1)) if m else -1


def file_url(folder, name):
    return f"/file?folder={quote(folder)}&name={quote(name)}"


def nice_step(span, count=5):
    """Round step (1, 2 or 5 times a power of 10) giving about count ticks."""
    raw = span / count
    power = 10 ** np.floor(np.log10(raw))
    return next(m * power for m in (1, 2, 5, 10) if m * power >= raw)


def line_chart(title, xs, values, labels, fmt, xlabel="epoch"):
    """Inline SVG line chart with crosshair tooltip and a table view.

    Args:
        title (str): chart title.
        xs, values (list): x and y of the points, x increasing.
        labels (list): optional per-point text shown in tooltip and table.
        fmt (callable): formats a y value.
        xlabel (str): name of the x axis.
    """
    if not values:
        return f"<h2>{html.escape(title)}</h2><p class=muted>No data.</p>"
    w, h, left, top, bottom = 720, 220, 48, 10, 26
    ph, pw = h - top - bottom, w - left - 10
    ymin = min(0.0, min(values))
    ystep = nice_step(max(max(values) - ymin, 1e-9))
    ymax = ymin + ystep * np.ceil((max(values) - ymin) / ystep + 1e-9)
    xmin, xmax = xs[0], max(xs[-1], xs[0] + 1)
    decimals = max(0, int(-np.floor(np.log10(ystep))))

    def sx(x):
        return left + (x - xmin) / (xmax - xmin) * pw

    def sy(y):
        return top + ph - (y - ymin) / (ymax - ymin) * ph

    pts = " ".join(f"{sx(x):.1f},{sy(v):.1f}" for x, v in zip(xs, values))
    grid = []
    for v in np.arange(ymin, ymax + ystep / 2, ystep):
        grid.append(
            f'<line x1="{left}" x2="{w - 10}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
            f'<text x="{left - 6}" y="{sy(v) + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="var(--text-2)">{v:.{decimals}f}</text>'
        )
    xstep = max(1, int(nice_step(xmax - xmin)))
    for x in range(int(np.ceil(xmin / xstep)) * xstep, int(xmax) + 1, xstep):
        grid.append(
            f'<text x="{sx(x):.1f}" y="{h - 8}" text-anchor="middle" '
            f'font-size="11" fill="var(--text-2)">{x}</text>'
        )
    data = [[x, v, fmt(v), labels[i] if i < len(labels) else ""]
            for i, (x, v) in enumerate(zip(xs, values))]
    geom = dict(w=w, left=left, top=top, h=ph, pw=pw, xmin=xmin, xmax=xmax,
                ymin=ymin, ymax=ymax, xlabel=xlabel)
    rows = "".join(
        f"<tr><td>{x}</td><td>{fmt(v)}</td><td>{html.escape(lab)}</td></tr>"
        for x, v, _, lab in data
    )
    marker = (f'<circle cx="{sx(xs[0]):.1f}" cy="{sy(values[0]):.1f}" r="4" '
              f'fill="var(--series-1)" stroke="var(--surface)" stroke-width="2"/>'
              if len(values) == 1 else "")
    return f"""
<h2>{html.escape(title)}</h2>
<div class="chart" data-points='{html.escape(json.dumps(data))}'
     data-geom='{html.escape(json.dumps(geom))}'>
  <svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}">
    {''.join(grid)}
    <polyline points="{pts}" fill="none" stroke="var(--series-1)"
      stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
    {marker}
    <line class="cross" y1="{top}" y2="{top + ph}" stroke="var(--text-2)"
      stroke-width="1" style="display:none"/>
    <circle class="hover-dot" r="4" fill="var(--series-1)"
      stroke="var(--surface)" stroke-width="2" style="display:none"/>
  </svg>
  <div class="tooltip"></div>
  <div class="muted" style="font-size:12px">x: {html.escape(xlabel)}</div>
</div>
<details><summary>Table</summary>
<table><tr><th>{html.escape(xlabel.capitalize())}</th><th>Value</th><th>Object</th></tr>{rows}</table>
</details>"""


def goals_rows(folder, names):
    rows = []
    for name in names:
        try:
            goals = np.load(os.path.join(folder, name), allow_pickle=True)[0]
            world = goals["world"][0] if goals["world"] else ""
            angle = goals["angle"][0] if goals["angle"] else ""
            seq = " ".join(
                "{},{}".format(*np.asarray(g).ravel()[:2].astype(int))
                for g in goals["goal"]
            )
            rows.append(
                f"<tr><td>{html.escape(name)}</td><td>{html.escape(str(world))}</td>"
                f"<td>{float(angle):.2f}</td><td>{len(goals['goal'])}</td>"
                f"<td style='font-family:monospace'>{html.escape(seq)}</td></tr>"
            )
        except Exception as e:
            rows.append(
                f"<tr><td>{html.escape(name)}</td><td colspan=4 class=muted>"
                f"unreadable: {html.escape(str(e))}</td></tr>"
            )
    return rows


def gallery(folder, names, link_gif=True):
    items = []
    for name in names:
        gif = name[:-4] + ".gif"
        target = gif if link_gif and os.path.exists(os.path.join(folder, gif)) else name
        items.append(
            f'<figure><a href="{file_url(folder, target)}" target="_blank">'
            f'<img loading="lazy" src="{file_url(folder, name)}" alt="{html.escape(name)}">'
            f"</a><figcaption>{html.escape(name)}</figcaption></figure>"
        )
    return f'<div class="gallery">{"".join(items)}</div>' if items else "<p class=muted>None.</p>"


def legacy_report(folder):
    """Report built from log, loaded_params and the files of the folder
    (simulations without data_sim)."""
    files = sorted(os.listdir(folder))
    comp, samples, labels = parse_log(folder)
    name = " ".join(read_lines(os.path.join(folder, "NAME"))) or os.path.basename(folder)

    tiles = [("Run", html.escape(name)), ("Epochs logged", str(len(samples)))]
    if comp:
        best = int(np.argmax(comp))
        tiles += [("Last competence", f"{comp[-1]:.3f}"),
                  ("Best competence", f"{comp[best]:.3f} (epoch {best})")]
    out = ['<div class="tiles">'] + [
        f'<div class="tile"><span class="muted">{k}</span><strong>{v}</strong></div>'
        for k, v in tiles
    ] + ["</div>"]

    out.append(line_chart("Competence", list(range(len(comp))), comp, labels,
                          lambda v: f"{v:.3f}"))
    out.append(line_chart("Salient samples per epoch", list(range(len(samples))),
                          [float(x) for x in samples], labels, lambda v: f"{v:.0f}"))

    params_file = ("final_parameters"
                   if os.path.isfile(os.path.join(folder, "final_parameters"))
                   else "loaded_params")
    params = read_lines(os.path.join(folder, params_file))
    rows = "".join(
        "<tr><td>{}</td><td>{}</td></tr>".format(
            *(html.escape(x.strip()) for x in line.split("=", 1))
        )
        for line in params if "=" in line
    )
    out.append("<h2>Parameters</h2>" + (
        f"<details><summary>{len(params)} parameters ({params_file})</summary>"
        f"<table>{rows}</table></details>" if rows else "<p class=muted>No loaded_params.</p>"))

    maps = sorted((f for f in files if re.match(r"maps_\d+\.png$", f)), key=epoch_of)
    out.append("<h2>Maps snapshots</h2><p class=muted>Click an image for the gif of "
               "the preceding epochs.</p>" + gallery(folder, maps))
    sims = [f for f in files if re.match(r"sim_\d+\.gif$", f)]
    if sims:
        out.append("<h2>Training simulations</h2>" + gallery(folder, sims, link_gif=False))

    goals = [f for f in files if f.startswith("goals") and f.endswith(".npy")]
    out.append("<h2>Test results</h2>")
    if goals:
        out.append("<table><tr><th>File</th><th>World</th><th>Angle</th>"
                   "<th>Saccades</th><th>Goal sequence (row,col)</th></tr>"
                   + "".join(goals_rows(folder, goals)) + "</table>")
    else:
        out.append("<p class=muted>No goals files (run scripts/tests.py).</p>")
    tests = [f for f in files if "_test_" in f and f.endswith(".gif")]
    if tests:
        out.append("<h3>Test animations</h3>" + gallery(folder, tests, link_gif=False))

    out.append(log_tails(folder))
    return "\n".join(out)


def log_tails(folder):
    """The last 40 lines of log and nohup.out, if present."""
    out = []
    for logname in ["log", "nohup.out"]:
        lines = read_lines(os.path.join(folder, logname))
        if lines:
            out.append(f"<h2>{logname} (last 40 lines)</h2>"
                       f"<pre>{html.escape(chr(10).join(lines[-40:]))}</pre>")
    return "\n".join(out)


DATA_DIR = "data_sim"
METRIC_TITLES = {
    "competence": "Competence",
    "salient_triangles": "Salient samples from triangle episodes",
    "salient_squares": "Salient samples from square episodes",
    "visual_conditions": "Weight change of the visual-conditions map",
    "visual_effects": "Weight change of the visual-effects map",
    "attention": "Weight change of the attention map",
}


def load_runs(folder):
    """Read every run of folder/data_sim (see src/local_wandb.py).

    Returns:
        list: dicts with "dir" (run folder name), "info" (config.json) and
        "rows" (metrics.jsonl rows), sorted by start time.
    """
    base = os.path.join(folder, DATA_DIR)
    runs = []
    for name in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        run_dir = os.path.join(base, name)
        try:
            with open(os.path.join(run_dir, "config.json")) as f:
                info = json.load(f)
        except (OSError, ValueError):
            continue
        rows = []
        for line in read_lines(os.path.join(run_dir, "metrics.jsonl")):
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
        runs.append(dict(dir=name, info=info, rows=rows))
    return sorted(runs, key=lambda r: r["info"].get("start_time") or 0)


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def merged_series(runs):
    """Numeric metrics of the given runs merged by step (later runs win).

    Returns:
        dict: key -> sorted list of (step, value).
    """
    series = {}
    for run in runs:
        for row in run["rows"]:
            for key, value in row.items():
                if not key.startswith("_") and is_number(value):
                    series.setdefault(key, {})[row["_step"]] = value
    return {k: sorted(v.items()) for k, v in series.items()}


def media_items(runs):
    """All logged media as (key, step, run dir, entry), sorted by step."""
    items = [
        (key, row["_step"], run["dir"], value)
        for run in runs
        for row in run["rows"]
        for key, value in row.items()
        if isinstance(value, dict) and value.get("_type") in ("image", "video")
    ]
    return sorted(items, key=lambda x: (x[0], x[1]))


def media_gallery(folder, items, links=None):
    """Gallery of media entries; links maps (step, run) to a click target."""
    figures = []
    for key, step, run_dir, entry in items:
        src = f"{DATA_DIR}/{run_dir}/{entry['path']}"
        target = (links or {}).get((step, run_dir), src)
        figures.append(
            f'<figure><a href="{file_url(folder, target)}" target="_blank">'
            f'<img loading="lazy" src="{file_url(folder, src)}" alt="{html.escape(key)}">'
            f"</a><figcaption>{html.escape(key)}, step {step}</figcaption></figure>"
        )
    return f'<div class="gallery">{"".join(figures)}</div>'


def when(timestamp):
    if not timestamp:
        return ""
    return datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def data_sim_report(folder, runs):
    """Report built from the runs logged in folder/data_sim."""
    train = [r for r in runs if r["info"].get("job_type") == "train"]
    tests = [r for r in runs if r["info"].get("job_type") == "test"]
    other = [r for r in runs if r not in train and r not in tests]
    series = merged_series(train + other)
    comp = series.get("competence", [])
    name = next((r["info"].get("name") for r in reversed(train) if r["info"].get("name")),
                os.path.basename(folder))

    tiles = [("Run", html.escape(str(name))), ("Training runs", str(len(train))),
             ("Test runs", str(len(tests))), ("Epochs logged", str(len(comp)))]
    if comp:
        best = max(comp, key=lambda sv: sv[1])
        tiles += [("Last competence", f"{comp[-1][1]:.3f}"),
                  ("Best competence", f"{best[1]:.3f} (epoch {best[0]})")]
    out = ['<div class="tiles">'] + [
        f'<div class="tile"><span class="muted">{k}</span><strong>{v}</strong></div>'
        for k, v in tiles
    ] + ["</div>"]

    rows = []
    for run in runs:
        info = run["info"]
        status = when(info.get("end_time")) or "running or interrupted"
        rows.append(
            f"<tr><td>{html.escape(run['dir'])}</td>"
            f"<td>{html.escape(str(info.get('job_type') or ''))}</td>"
            f"<td>{html.escape(str(info.get('name') or ''))}</td>"
            f"<td>{when(info.get('start_time'))}</td><td>{html.escape(status)}</td>"
            f"<td>{len(run['rows'])}</td></tr>"
        )
    out.append("<h2>Runs (data_sim)</h2><table><tr><th>Folder</th><th>Job</th>"
               "<th>Name</th><th>Started</th><th>Finished</th><th>Rows</th></tr>"
               + "".join(rows) + "</table>")

    tri = dict(series.get("salient_triangles", []))
    sq = dict(series.get("salient_squares", []))

    def label(step):
        if step not in tri and step not in sq:
            return ""
        return "triangle" if tri.get(step, 0) >= sq.get(step, 0) else "square"

    if tri or sq:
        steps = sorted(set(tri) | set(sq))
        out.append(line_chart(
            "Salient samples per epoch", steps,
            [float(tri.get(s, 0) + sq.get(s, 0)) for s in steps],
            [label(s) for s in steps], lambda v: f"{v:.0f}"))
    keys = [k for k in METRIC_TITLES if k in series and not k.startswith("salient_")]
    keys += sorted(k for k in series if k not in METRIC_TITLES)
    for key in keys:
        steps, values = zip(*series[key])
        fmt = (lambda v: f"{v:.0f}") if all(float(v).is_integer() for v in values) \
            else (lambda v: f"{v:.3f}")
        out.append(line_chart(METRIC_TITLES.get(key, key), list(steps),
                              [float(v) for v in values],
                              [label(s) for s in steps], fmt))

    media = media_items(train + other)
    history = {(step, run_dir): f"{DATA_DIR}/{run_dir}/{entry['path']}"
               for key, step, run_dir, entry in media if key == "history"}
    last = [m for m in media if m[0] == "last"]
    if last:
        out.append("<h2>Maps snapshots</h2><p class=muted>Click an image for the "
                   "gif of the preceding epochs.</p>"
                   + media_gallery(folder, last, history))
    for key in sorted({m[0] for m in media} - {"last", "history"}):
        out.append(f"<h2>{html.escape(key)}</h2>"
                   + media_gallery(folder, [m for m in media if m[0] == key]))

    if tests:
        out.append("<h2>Test runs</h2>")
        for run in tests:
            cfg = run["info"].get("config", {})
            pose = cfg.get("test_object_params") or {}
            goals_file = cfg.get("goals_file", "")
            goals = (goals_rows(folder, [goals_file])
                     if goals_file and os.path.isfile(os.path.join(folder, goals_file))
                     else [])
            out.append(
                f"<h3>{html.escape(run['dir'])}</h3><p class=muted>"
                f"world {html.escape(str(cfg.get('test_world')))}, "
                f"position {html.escape(str(pose.get('pos')))}, "
                f"rotation {html.escape(str(pose.get('rot')))}, "
                f"started {when(run['info'].get('start_time'))}</p>")
            if goals:
                out.append("<table><tr><th>File</th><th>World</th><th>Angle</th>"
                           "<th>Saccades</th><th>Goal sequence (row,col)</th></tr>"
                           + "".join(goals) + "</table>")
            test_media = media_items([run])
            if test_media:
                out.append(media_gallery(folder, test_media))

    source = train[-1] if train else runs[-1]
    cfg = source["info"].get("config", {})
    rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(json.dumps(v))}</td></tr>"
        for k, v in sorted(cfg.items())
    )
    out.append("<h2>Configuration</h2>" + (
        f"<details><summary>{len(cfg)} parameters ({html.escape(source['dir'])})"
        f"</summary><table>{rows}</table></details>" if rows
        else "<p class=muted>No configuration.</p>"))
    out.append(log_tails(folder))
    return "\n".join(out)


def folder_report(folder):
    """data_sim report if the folder has logged runs, else the legacy one."""
    runs = load_runs(folder)
    return data_sim_report(folder, runs) if runs else legacy_report(folder)


IMAGE_TYPES = {".png": "image/png", ".gif": "image/gif"}


def is_sim_folder(folder):
    """True if folder contains a log or loaded_params file, or data_sim."""
    return os.path.isdir(os.path.join(folder, DATA_DIR)) or any(
        os.path.isfile(os.path.join(folder, f)) for f in ["log", "loaded_params"]
    )


def count_sim_folders(folder):
    """Number of simulation folders directly inside folder."""
    try:
        names = os.listdir(folder)
    except OSError:
        return 0
    return sum(
        is_sim_folder(os.path.join(folder, n))
        for n in names
        if not n.startswith(".") and os.path.isdir(os.path.join(folder, n))
    )


def browse(root, path):
    """List the subfolders of path, which must lie inside root.

    Paths are compared as written (normalized, symlinks not resolved), so
    symlinked folders below root can be browsed, while ".." cannot leave
    root. Returns a JSON-serializable dict with the path, its parent (None
    at root), whether path is a simulation folder, and its subfolders
    ({name, path, sim, sims}, sims = number of simulation folders directly
    inside); paths outside root fall back to root.
    """
    root = os.path.normpath(os.path.abspath(root))
    path = os.path.normpath(os.path.abspath(path)) if path else root
    if not os.path.isdir(path) or os.path.commonpath([root, path]) != root:
        path = root
    dirs = []
    try:
        names = sorted(os.listdir(path))
    except OSError as e:
        return dict(path=path, parent=None, sim=False, dirs=[], error=str(e))
    for name in names:
        full = os.path.join(path, name)
        if (name.startswith(".") or name in ("wandb", DATA_DIR)
                or not os.path.isdir(full)):
            continue
        dirs.append(dict(name=name, path=full, sim=is_sim_folder(full),
                         sims=count_sim_folders(full)))
    parent = None if path == root else os.path.dirname(path)
    return dict(path=path, parent=parent, sim=is_sim_folder(path), dirs=dirs)


def candidate_folders(root):
    """Simulation folders up to two levels below root (symlinks followed;
    the search does not descend into simulation folders other than root)."""
    found = []
    for base, dirs, filenames in os.walk(root, followlinks=True):
        depth = os.path.relpath(base, root).count(os.sep)
        if depth >= 2 or (base != root and is_sim_folder(base)):
            dirs[:] = []
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "wandb"]
        if is_sim_folder(base):
            found.append(os.path.abspath(base))
    return sorted(found)


class Handler(SimpleHTTPRequestHandler):
    root = os.getcwd()
    default_folder = ""

    def send_text(self, body, status=200, ctype="text/html; charset=utf-8"):
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        raw = query.get("folder", [self.default_folder])[0]
        folder = os.path.abspath(os.path.expanduser(raw)) if raw else ""
        if url.path == "/browse":
            return self.send_text(
                json.dumps(browse(self.root, query.get("path", [""])[0])),
                ctype="application/json",
            )
        if url.path == "/file":
            return self.send_file(folder, query.get("name", [""])[0])
        if url.path != "/":
            return self.send_text("Not found", 404, "text/plain")

        options = "".join(
            f'<option value="{html.escape(f)}">' for f in candidate_folders(self.root)
        )
        body = ""
        if folder:
            if is_sim_folder(folder):
                body = folder_report(folder)
            else:
                body = (f"<p>Not a simulation folder (no data_sim, log or "
                        f"loaded_params): "
                        f"{html.escape(folder)}</p>")
        self.send_text(f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Simulation results</title><style>{STYLE}</style></head>
<body><main>
<h1>Simulation results</h1>
<form method="get" action="/">
  <input type="text" name="folder" list="folders" value="{html.escape(folder)}"
    placeholder="Path to a simulation folder">
  <datalist id="folders">{options}</datalist>
  <button type="button" id="browse">Browse...</button>
  <button type="submit">Show</button>
</form>
{body}
</main>
<dialog id="browser" aria-label="Choose a simulation folder">
  <div class="dlg-head"></div><ul class="dlg-list"></ul>
</dialog>
<script>{CHART_JS}{BROWSE_JS}</script></body></html>""")

    def send_file(self, folder, name):
        """Serve an image directly inside folder or anywhere under
        folder/data_sim."""
        ctype = IMAGE_TYPES.get(os.path.splitext(name)[1].lower())
        if not folder or not name or ctype is None or not is_sim_folder(folder):
            return self.send_text("Not found", 404, "text/plain")
        real_folder = os.path.realpath(folder)
        data_dir = os.path.join(real_folder, DATA_DIR)
        path = os.path.realpath(os.path.join(real_folder, name))
        allowed = os.path.dirname(path) == real_folder or (
            os.path.commonpath([data_dir, path]) == data_dir
        )
        if not allowed or not os.path.isfile(path):
            return self.send_text("Not found", 404, "text/plain")
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def default_host():
    """This machine's Tailscale IPv4, or 127.0.0.1 if it cannot be found."""
    try:
        out = subprocess.run(
            ["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5
        )
        address = out.stdout.split()[0] if out.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired, IndexError):
        address = ""
    return address or "127.0.0.1"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", nargs="?", default="",
                        help="Simulation folder shown first.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--host",
        default=None,
        help="Address to listen on (default: Tailscale IPv4, else 127.0.0.1).",
    )
    args = parser.parse_args()

    host = args.host or default_host()
    Handler.default_folder = os.path.abspath(args.folder) if args.folder else ""
    server = HTTPServer((host, args.port), Handler)
    print(f"Serving on http://{host}:{args.port} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
