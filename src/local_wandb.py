"""Offline stand-in for the part of the wandb API used by the simulations.

Every `init` creates a run folder ./data_sim/run-<timestamp>/ containing:
    - config.json: project, entity, name, job_type, start/end time and the
      run config (the simulation parameters);
    - metrics.jsonl: one JSON object per `log` call, with "_step",
      "_timestamp" and the logged values; scalars (numbers, numpy/torch
      scalars) become plain numbers, and media become
      {"_type": "image"|"video", "path": "media/<file>"};
    - summary.json: the last logged value of every key;
    - media/: the logged images and videos (hard links to the original
      files when possible, copies otherwise).

Use `backend(use_wandb)` to get either the real wandb module or this one.
"""
import datetime
import json
import os
import re
import shutil
import sys
import time

import numpy as np


DATA_DIR = "data_sim"
_run = None


class _Media:
    """A file to be stored with the run; `kind` is "image" or "video"."""

    kind = None

    def __init__(self, data_or_path, caption=None, format=None, **kwargs):
        """
        Args:
            data_or_path (str): path of an existing image or video file.
            caption (str, optional): stored with the media entry.
            format (str, optional): file extension used if the path has none.
        """
        self.path = data_or_path
        self.caption = caption
        self.format = format


class Image(_Media):
    """An image file, as wandb.Image(path)."""

    kind = "image"


class Video(_Media):
    """A video or animated gif file, as wandb.Video(path, format=...)."""

    kind = "video"


def _jsonable(value):
    """Convert numbers, numpy/torch values and containers to JSON types."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.item() if value.size == 1 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


class Run:
    """A local run writing to ./data_sim/run-<timestamp>/."""

    def __init__(
        self,
        project=None,
        entity=None,
        name=None,
        config=None,
        job_type=None,
        dir=None,
        **kwargs,
    ):
        """
        Args:
            project, entity, name, job_type: stored in config.json.
            config (dict, optional): run configuration.
            dir (str, optional): parent of data_sim (default: cwd).
            **kwargs: other wandb.init arguments, ignored.
        """
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.dir = os.path.join(dir or ".", DATA_DIR, f"run-{stamp}")
        os.makedirs(os.path.join(self.dir, "media"))
        self.name = name
        self.summary = {}
        self._step = 0
        self._info = dict(
            project=project,
            entity=entity,
            name=name,
            job_type=job_type,
            start_time=time.time(),
            end_time=None,
            config=_jsonable(config or {}),
        )
        self._write_json("config.json", self._info)
        self._metrics = open(os.path.join(self.dir, "metrics.jsonl"), "a")

    def _write_json(self, filename, data):
        with open(os.path.join(self.dir, filename), "w") as f:
            json.dump(data, f, indent=2)

    def _save_media(self, key, media, step):
        extension = os.path.splitext(media.path)[1] or f".{media.format}"
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
        filename = f"{slug}_{step:06d}{extension}"
        target = os.path.join(self.dir, "media", filename)
        if os.path.exists(target):
            os.remove(target)
        try:
            os.link(media.path, target)
        except OSError:
            shutil.copy2(media.path, target)
        entry = {"_type": media.kind, "path": f"media/{filename}"}
        if media.caption:
            entry["caption"] = media.caption
        return entry

    def log(self, data, step=None, commit=True):
        """Append a metrics row; step defaults to the next step, as in wandb.

        Args:
            data (dict): values to log (scalars, arrays, Image, Video).
            step (int, optional): step of the row.
            commit (bool): advance the default step after this row.
        """
        step = self._step if step is None else int(step)
        row = {"_step": step, "_timestamp": time.time()}
        for key, value in data.items():
            if isinstance(value, _Media):
                row[key] = self._save_media(key, value, step)
            else:
                row[key] = _jsonable(value)
        self._metrics.write(json.dumps(row) + "\n")
        self._metrics.flush()
        self.summary.update({k: v for k, v in row.items() if not k.startswith("_")})
        if commit:
            self._step = step + 1

    def finish(self):
        """Write summary.json and the end time, and close the run."""
        self._info["end_time"] = time.time()
        self._write_json("config.json", self._info)
        self._write_json("summary.json", self.summary)
        self._metrics.close()


def init(**kwargs):
    """Start a run (see Run) and make it the current one."""
    global _run
    if _run is not None:
        _run.finish()
    _run = Run(**kwargs)
    return _run


def log(data, step=None, commit=True):
    """Log to the current run (see Run.log)."""
    if _run is None:
        raise RuntimeError("local_wandb.log called before local_wandb.init")
    _run.log(data, step=step, commit=commit)


def finish():
    """Finish the current run, if any."""
    global _run
    if _run is not None:
        _run.finish()
        _run = None


def backend(use_wandb):
    """Return the real wandb module if use_wandb, otherwise this module."""
    if use_wandb:
        import wandb

        return wandb
    return sys.modules[__name__]
