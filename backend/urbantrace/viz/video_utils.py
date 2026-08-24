"""Video helpers for notebook playback.

OpenCV writes mp4v, which most browsers refuse to play inside a notebook.
`to_h264` re-encodes with the ffmpeg binary bundled by imageio-ffmpeg, so the
annotated output plays inline with no system ffmpeg install.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def to_h264(src, dst=None, crf: int = 26, max_width: int | None = 1280):
    """Re-encode to browser-playable H.264. Returns the output path."""
    src = Path(src)
    dst = Path(dst) if dst else src.with_name(src.stem + "_h264.mp4")
    if not src.exists():
        raise FileNotFoundError(src)

    vf = f"scale='min({max_width},iw)':-2" if max_width else "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
           "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
           "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           str(dst)]
    subprocess.run(cmd, check=True)
    return dst


def show_video(path, width: int = 900, autoplay: bool = False):
    """Return an IPython video element, re-encoding to H.264 if needed."""
    from IPython.display import HTML
    import base64

    path = Path(path)
    if "_h264" not in path.stem:
        h264 = path.with_name(path.stem + "_h264.mp4")
        path = h264 if h264.exists() else to_h264(path)

    data = base64.b64encode(path.read_bytes()).decode("ascii")
    attrs = "controls" + (" autoplay muted loop" if autoplay else "")
    return HTML(f'<video width="{width}" {attrs} '
                f'src="data:video/mp4;base64,{data}"></video>')
