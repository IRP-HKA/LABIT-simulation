"""
Build a composite video for a LabIT benchmark trial.

For every primitive in chronological order the output places the simulation
camera view on the left and an animated force profile (Fx, Fy, Fz) on the
right.  A yellow cursor on the force plot tracks the current simulation time
so that force events are visually aligned with the corresponding video frame.

Usage
-----
    python src/evaluation/make_trial_video.py <trial_folder> [options]

    trial_folder  – e.g. workcells/labit_benchmark/results/2026_05_21_17_07_46_trial_7_rnd

Options
-------
    --camera    Camera view to embed (default: default_view | top_view)
    --fps       Output frame rate – should match the recorded fps (default: 24)
    --out       Output file path   (default: <trial_folder>/trial_composite.mp4)
    --force-lim Maximum y-axis force limit in N; 0 = auto (default: 0)
"""

import argparse
import os
import re
import sys
from glob import glob

import numpy as np
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# ── naming helpers ────────────────────────────────────────────────────────────

_TS_RE = re.compile(r"^(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})")


def _ts_key(path: str) -> str:
    m = _TS_RE.match(os.path.basename(path.rstrip(os.sep)))
    return m.group(1) if m else ""


def _strip_ts(name: str) -> str:
    """Remove leading timestamp from a folder/file basename."""
    return _TS_RE.sub("", name).lstrip("_")


def _primitive_action(stem: str) -> str:
    """Extract the last token (action name) from a file stem, ignoring camera suffix."""
    # stem may end with _{camera} which was already stripped before calling this
    return stem.split("_")[-1]


# ── data collection ───────────────────────────────────────────────────────────

def collect_primitives(trial_dir: str, camera: str) -> list[tuple[str, str, str, str]]:
    """
    Return a list of (npz_path, video_path, subtask_label, action_label) tuples
    sorted chronologically across the whole trial.

    Only primitives that have both a .npz file and the requested camera .mp4 are
    included.
    """
    primitives = []
    subtask_dirs = sorted(glob(os.path.join(trial_dir, "*" + os.sep)), key=_ts_key)

    for sdir in subtask_dirs:
        subtask_label = _strip_ts(os.path.basename(sdir.rstrip(os.sep)))
        npz_files = sorted(glob(os.path.join(sdir, "*.npz")), key=_ts_key)

        for npz in npz_files:
            stem = npz[:-4]
            video = f"{stem}_{camera}.mp4"
            if not os.path.exists(video):
                continue
            action = _primitive_action(_strip_ts(os.path.basename(stem)))
            primitives.append((npz, video, subtask_label, action))

    return primitives


# ── force data ────────────────────────────────────────────────────────────────

def load_force(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps [N], forces [N×3]) with timestamps relative to primitive start."""
    d = np.load(npz_path)
    ts = np.asarray(d["timestamp"], dtype=float)
    ft = np.asarray(d["eef_fts"], dtype=float)
    if ft.ndim == 1:
        ft = ft[:, None]
    ft = ft[:, :3]          # Fx, Fy, Fz
    ts = ts - ts[0]         # make relative so frame 0 → t = 0
    return ts, ft


# ── matplotlib force panel ────────────────────────────────────────────────────

_FORCE_LABELS  = ["Fx [N]", "Fy [N]", "Fz [N]"]
_FORCE_COLORS  = ["#e74c3c", "#2ecc71", "#3498db"]
_BG_DARK       = "#0e0e1a"
_FG            = "white"
_GRID          = "#2a2a3a"

_ACTION_COLORS = {
    "moving":    "#4CAF50",
    "grasping":  "#2196F3",
    "inserting": "#FF5722",
    "screwing":  "#9C27B0",
}


def _make_force_panel(ts: np.ndarray, ft: np.ndarray,
                      panel_w: int, panel_h: int, dpi: int,
                      subtask_label: str, action: str,
                      force_lim: float) -> tuple:
    """
    Create a matplotlib figure showing the force profile for one primitive.
    Returns (fig, canvas, axes, cursor_lines).
    Call _update_cursor(cursor_lines, t) then _render_panel(canvas) per frame.
    """
    fig, axes = plt.subplots(3, 1,
                             figsize=(panel_w / dpi, panel_h / dpi),
                             dpi=dpi)
    fig.patch.set_facecolor(_BG_DARK)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.88, bottom=0.10, hspace=0.5)

    action_color = _ACTION_COLORS.get(action, "#FFFFFF")
    title = f"{subtask_label}\n{action}"
    fig.suptitle(title, color=action_color, fontsize=8, fontweight="bold",
                 x=0.6, y=0.97, va="top", linespacing=1.4)

    t_end = ts[-1] if len(ts) > 1 else 1.0
    ylim  = force_lim if force_lim > 0 else max(float(np.abs(ft).max()) * 1.3, 5.0)

    cursor_lines = []
    for i, (ax, label, color) in enumerate(zip(axes, _FORCE_LABELS, _FORCE_COLORS)):
        ax.set_facecolor(_BG_DARK)
        ax.plot(ts, ft[:, i], color=color, lw=0.8, alpha=0.9)
        ax.set_xlim(0.0, t_end)
        ax.set_ylim(-ylim, ylim)
        ax.set_ylabel(label, color=_FG, fontsize=7, labelpad=2)
        ax.tick_params(colors=_FG, labelsize=5.5)
        ax.axhline(0, color=_GRID, lw=0.5)
        ax.grid(True, alpha=0.25, color=_GRID, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID)
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("t [s]", color=_FG, fontsize=7)
        cl = ax.axvline(0.0, color="yellow", lw=1.2, alpha=0.85, zorder=5)
        cursor_lines.append(cl)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()   # initial draw of static content
    return fig, canvas, axes, cursor_lines


def _update_cursor(cursor_lines: list, t: float) -> None:
    for cl in cursor_lines:
        cl.set_xdata([t, t])


def _render_panel(canvas, panel_w: int, panel_h: int) -> np.ndarray:
    """Render the canvas to an (H, W, 3) uint8 array."""
    canvas.draw()
    buf = canvas.buffer_rgba()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(
        canvas.get_width_height()[::-1] + (4,))
    arr = arr[:, :, :3]
    if arr.shape[0] != panel_h or arr.shape[1] != panel_w:
        from PIL import Image
        arr = np.array(
            Image.fromarray(arr).resize((panel_w, panel_h), Image.LANCZOS))
    return arr


# ── text overlay ──────────────────────────────────────────────────────────────

def _add_label(frame: np.ndarray, subtask: str, action: str) -> np.ndarray:
    """Burn a small text label into the top-left corner of a video frame."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        color = _ACTION_COLORS.get(action, "#FFFFFF")
        text = f"{subtask}  ›  {action}"
        draw.rectangle([4, 4, len(text) * 6 + 8, 20], fill=(0, 0, 0, 160))
        draw.text((6, 5), text, fill=color)
        return np.array(img)
    except Exception:
        return frame


# ── main builder ──────────────────────────────────────────────────────────────

def build(trial_dir: str, camera: str, fps: int, out_path: str, force_lim: float) -> None:
    PANEL_W = 640
    DPI = 100

    primitives = collect_primitives(trial_dir, camera)
    if not primitives:
        print(f"No primitives found in {trial_dir!r} for camera '{camera}'.")
        print("Check that the trial folder contains subtask sub-folders with "
              f"*_{camera}.mp4 files.")
        sys.exit(1)

    n_subtasks = len(set(p[2] for p in primitives))
    print(f"Found {len(primitives)} primitives across {n_subtasks} subtasks.")

    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=None)

    for idx, (npz_path, video_path, subtask, action) in enumerate(primitives):
        print(f"  [{idx+1:3d}/{len(primitives)}]  {subtask} / {action}")

        if os.path.getsize(npz_path) == 0:
            print("         (empty .npz – skipping)")
            continue

        ts, ft = load_force(npz_path)
        t_end = ts[-1] if len(ts) > 1 else 1.0

        reader = imageio.get_reader(video_path)

        fig = canvas = axes = cursor_lines = None
        panel_h = None

        for frame_i, raw_frame in enumerate(reader):
            vframe = np.asarray(raw_frame)[:, :, :3]   # ensure RGB
            vh, vw = vframe.shape[:2]

            # Create panel figure once we know the video height
            if fig is None:
                panel_h = vh
                fig, canvas, axes, cursor_lines = _make_force_panel(
                    ts, ft, PANEL_W, panel_h, DPI, subtask, action, force_lim)

            # Cursor time: frame_i / fps seconds elapsed in this primitive
            cursor_t = min(frame_i / fps, t_end)
            _update_cursor(cursor_lines, cursor_t)
            panel_frame = _render_panel(canvas, PANEL_W, panel_h)

            labeled_frame = _add_label(vframe, subtask, action)
            composite = np.concatenate([labeled_frame, panel_frame], axis=1)
            writer.append_data(composite)

        reader.close()
        if fig is not None:
            plt.close(fig)

    writer.close()
    print(f"\nDone. Composite video → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a composite LabIT trial video with aligned force plots.")
    parser.add_argument("trial_dir",
                        help="Path to the trial results folder")
    parser.add_argument("--camera", default="default_view",
                        help="Camera name in video filenames (default: default_view)")
    parser.add_argument("--fps", type=int, default=24,
                        help="Output frame rate (default: 24)")
    parser.add_argument("--out", default=None,
                        help="Output path (default: <trial_dir>/trial_composite.mp4)")
    parser.add_argument("--force-lim", type=float, default=0.0,
                        help="Force axis limit in N; 0 = auto per primitive (default: 0)")
    args = parser.parse_args()

    trial_dir = os.path.realpath(args.trial_dir)
    if not os.path.isdir(trial_dir):
        print(f"Error: '{trial_dir}' is not a directory.")
        sys.exit(1)

    out_path = args.out or os.path.join(trial_dir, "trial_composite.mp4")
    build(trial_dir, args.camera, args.fps, out_path, args.force_lim)


if __name__ == "__main__":
    main()
