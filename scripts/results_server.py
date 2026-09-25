"""Local web server that shows all the results of a simulation folder.

Usage:
    python /path/to/scripts/results_server.py [FOLDER] [--port 8000]
        [--host 127.0.0.1]

Open http://127.0.0.1:8000 and type (or pick) a simulation folder in the
form, or press "Browse..." to navigate the folders below the launch
directory in a popup (simulation folders have a Select button). FOLDER, if
given, is shown first; the folder list offers every directory under the
launch directory (two levels deep) containing a log or loaded_params. To
reach the page from other machines, pass --host with an address of this
machine (for example its Tailscale IP).

For the chosen folder the page shows:
    - a summary (NAME, epochs run, last and best competence);
    - competence and salient samples per epoch, parsed from `log`
      ("comp:" lines are written only when main.py ran with -w);
    - the parameters in loaded_params;
    - the maps snapshots (maps_*.png, click for the gif) and sim_*.gif;
    - the test results: one row per goals*.npy (world, angle, goal
      sequence) and the *_test_* gifs;
    - the last lines of log and nohup.out.

Only simulation folders (containing log or loaded_params) are shown, only
images (png/gif) directly inside them are served, and the server listens on
localhost by default.
"""
import argparse
import html
import json
import os
import re
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
      a.textContent = x.name + '/'; a.onclick = function () { load(x.path); };
      li.appendChild(a);
      if (x.sim) {
        var b = document.createElement('span'); b.className = 'badge';
        b.textContent = 'simulation'; li.appendChild(b);
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
  function show(evt) {
    var r = svg.getBoundingClientRect();
    var x = (evt.clientX - r.left) * g.w / r.width;
    var i = Math.round((x - g.left) / g.dx);
    i = Math.max(0, Math.min(data.length - 1, i));
    var p = data[i], px = g.left + i * g.dx;
    var py = g.top + g.h - (p[1] - g.ymin) / (g.ymax - g.ymin) * g.h;
    line.setAttribute('x1', px); line.setAttribute('x2', px);
    dot.setAttribute('cx', px); dot.setAttribute('cy', py);
    line.style.display = dot.style.display = '';
    tip.textContent = '';
    var b = document.createElement('strong'); b.textContent = p[2];
    tip.appendChild(b);
    tip.appendChild(document.createTextNode('  epoch ' + p[0] + (p[3] ? '  ' + p[3] : '')));
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


def line_chart(title, values, labels, fmt):
    """Inline SVG line chart with crosshair tooltip and a table view."""
    if not values:
        return f"<h2>{html.escape(title)}</h2><p class=muted>No data in log.</p>"
    w, h, left, top, bottom = 720, 220, 48, 10, 26
    ph = h - top - bottom
    ymin = min(0.0, min(values))
    ystep = nice_step(max(max(values) - ymin, 1e-9))
    ymax = ymin + ystep * np.ceil((max(values) - ymin) / ystep + 1e-9)
    n = len(values)
    dx = (w - left - 10) / max(n - 1, 1)
    pts = " ".join(
        f"{left + i * dx:.1f},{top + ph - (v - ymin) / (ymax - ymin) * ph:.1f}"
        for i, v in enumerate(values)
    )
    grid = []
    for v in np.arange(ymin, ymax + ystep / 2, ystep):
        y = top + ph - (v - ymin) / (ymax - ymin) * ph
        grid.append(
            f'<line x1="{left}" x2="{w - 10}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
            f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="var(--text-2)">{fmt(v)}</text>'
        )
    xstep = max(1, int(nice_step(max(n - 1, 1))))
    for i in range(0, n, xstep):
        grid.append(
            f'<text x="{left + i * dx:.1f}" y="{h - 8}" text-anchor="middle" '
            f'font-size="11" fill="var(--text-2)">{i}</text>'
        )
    data = [[i, v, fmt(v), labels[i] if i < len(labels) else ""]
            for i, v in enumerate(values)]
    geom = dict(w=w, left=left, top=top, h=ph, dx=dx, ymin=ymin, ymax=ymax)
    rows = "".join(
        f"<tr><td>{i}</td><td>{fmt(v)}</td><td>{html.escape(lab)}</td></tr>"
        for i, v, _, lab in data
    )
    return f"""
<h2>{html.escape(title)}</h2>
<div class="chart" data-points='{html.escape(json.dumps(data))}'
     data-geom='{json.dumps(geom)}'>
  <svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}">
    {''.join(grid)}
    <polyline points="{pts}" fill="none" stroke="var(--series-1)"
      stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
    <line class="cross" y1="{top}" y2="{top + ph}" stroke="var(--text-2)"
      stroke-width="1" style="display:none"/>
    <circle class="hover-dot" r="4" fill="var(--series-1)"
      stroke="var(--surface)" stroke-width="2" style="display:none"/>
  </svg>
  <div class="tooltip"></div>
  <div class="muted" style="font-size:12px">x: epoch</div>
</div>
<details><summary>Table</summary>
<table><tr><th>Epoch</th><th>Value</th><th>Object</th></tr>{rows}</table>
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


def folder_report(folder):
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

    out.append(line_chart("Competence", comp, labels, lambda v: f"{v:.3f}"))
    out.append(line_chart("Salient samples per epoch", [float(x) for x in samples],
                          labels, lambda v: f"{v:.0f}"))

    params = read_lines(os.path.join(folder, "loaded_params"))
    rows = "".join(
        "<tr><td>{}</td><td>{}</td></tr>".format(
            *(html.escape(x.strip()) for x in line.split("=", 1))
        )
        for line in params if "=" in line
    )
    out.append("<h2>Parameters</h2>" + (
        f"<details><summary>{len(params)} parameters (loaded_params)</summary>"
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
        out.append("<p class=muted>No goals files (run scripts/tests.sh).</p>")
    tests = [f for f in files if "_test_" in f and f.endswith(".gif")]
    if tests:
        out.append("<h3>Test animations</h3>" + gallery(folder, tests, link_gif=False))

    for logname in ["log", "nohup.out"]:
        lines = read_lines(os.path.join(folder, logname))
        if lines:
            out.append(f"<h2>{logname} (last 40 lines)</h2>"
                       f"<pre>{html.escape(chr(10).join(lines[-40:]))}</pre>")
    return "\n".join(out)


IMAGE_TYPES = {".png": "image/png", ".gif": "image/gif"}


def is_sim_folder(folder):
    """True if folder contains a log or loaded_params file."""
    return any(
        os.path.isfile(os.path.join(folder, f)) for f in ["log", "loaded_params"]
    )


def browse(root, path):
    """List the subfolders of path, which must lie inside root.

    Returns a JSON-serializable dict with the absolute path, its parent
    (None at root), whether path is a simulation folder, and its
    subfolders ({name, path, sim}); paths outside root fall back to root.
    """
    root = os.path.realpath(root)
    path = os.path.realpath(path) if path else root
    if not os.path.isdir(path) or os.path.commonpath([root, path]) != root:
        path = root
    dirs = []
    try:
        names = sorted(os.listdir(path))
    except OSError as e:
        return dict(path=path, parent=None, sim=False, dirs=[], error=str(e))
    for name in names:
        full = os.path.join(path, name)
        if name.startswith(".") or name == "wandb" or not os.path.isdir(full):
            continue
        dirs.append(dict(name=name, path=full, sim=is_sim_folder(full)))
    parent = None if path == root else os.path.dirname(path)
    return dict(path=path, parent=parent, sim=is_sim_folder(path), dirs=dirs)


def candidate_folders(root):
    found = []
    for base, dirs, filenames in os.walk(root):
        depth = os.path.relpath(base, root).count(os.sep)
        if depth >= 2:
            dirs[:] = []
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "wandb"]
        if "log" in filenames or "loaded_params" in filenames:
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
                body = (f"<p>Not a simulation folder (no log or loaded_params): "
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
        path = os.path.join(folder, name)
        ctype = IMAGE_TYPES.get(os.path.splitext(name)[1].lower())
        if (not folder or not name or os.path.basename(name) != name
                or ctype is None or not is_sim_folder(folder)
                or not os.path.isfile(path)):
            return self.send_text("Not found", 404, "text/plain")
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", nargs="?", default="",
                        help="Simulation folder shown first.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    Handler.default_folder = os.path.abspath(args.folder) if args.folder else ""
    server = HTTPServer((args.host, args.port), Handler)
    print(f"Serving on http://{args.host}:{args.port} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
