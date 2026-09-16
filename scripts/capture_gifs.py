"""Captures a handful of deterministic states from the two interactive explorer pages
via headless Chrome screenshots, then stitches each set into a looping GIF. Uses the
`?reveal=1`/`?example=N` and `?frame=N` query-param hooks those pages expose
specifically for this (see the bottom of each page's <script>), not real click/wait
automation -- there is no browser-automation dependency, only a local Chrome install.

Each page is captured at 2x device-scale-factor into a generously tall window, then
trimmed to the real content height (chrome's --screenshot flag captures the full
window size regardless of how much of it the page actually fills, which otherwise
leaves a block of solid background color at the bottom of every frame).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXPLORER = ROOT / "explorer"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

PAGE_BG = (11, 14, 19)  # #0b0e13, both pages' --bg

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


def find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    found = shutil.which("chrome") or shutil.which("google-chrome")
    if found:
        return found
    raise RuntimeError("no Chrome install found; edit CHROME_CANDIDATES")


def screenshot(chrome: str, url: str, out_path: Path, window: str, user_data_dir: str) -> None:
    subprocess.run(
        [
            chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
            "--force-device-scale-factor=2",
            f"--user-data-dir={user_data_dir}",
            f"--screenshot={out_path}", f"--window-size={window}", url,
        ],
        check=True, capture_output=True, timeout=30,
    )


def content_height(image: Image.Image, bg: tuple[int, int, int], tol: int = 12) -> int:
    """The row index (from the top) below which every row is within `tol` of `bg` --
    i.e. how tall the actual rendered content is before the page's own background
    takes over, used to crop off the dead space a fixed capture window leaves."""
    arr = np.asarray(image.convert("RGB"))
    diff = np.abs(arr.astype(int) - np.array(bg)).max(axis=2)
    row_has_content = (diff > tol).any(axis=1)
    content_rows = np.nonzero(row_has_content)[0]
    if len(content_rows) == 0:
        return image.height
    return int(content_rows[-1]) + 1


def make_gif(frame_paths: list[Path], out_path: Path, durations_ms: list[int],
             bg: tuple[int, int, int] = PAGE_BG) -> None:
    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    pad = 24
    height = min(max(content_height(f, bg) for f in frames) + pad, frames[0].height)
    cropped = [f.crop((0, 0, f.width, height)) for f in frames]
    cropped[0].save(out_path, save_all=True, append_images=cropped[1:], duration=durations_ms, loop=0)


def main() -> None:
    chrome = find_chrome()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        user_data_dir = str(tmp_path / "chrome-profile")

        interrogation_url_base = (EXPLORER / "interrogation-room.html").as_uri()
        room_frames = []
        for i, reveal in enumerate((0, 1, 0)):
            out = tmp_path / f"room_{i}.png"
            screenshot(chrome, f"{interrogation_url_base}?example=0&reveal={reveal}",
                       out, "1300,2200", user_data_dir)
            room_frames.append(out)
        make_gif(room_frames, ASSETS / "interrogation_room.gif",
                  durations_ms=[1400, 2600, 900])
        print(f"wrote {ASSETS / 'interrogation_room.gif'}")

        shortcut_url_base = (EXPLORER / "shortcut.html").as_uri()
        shortcut_frames = []
        for i, frame in enumerate((0, 1, 2)):
            out = tmp_path / f"shortcut_{i}.png"
            screenshot(chrome, f"{shortcut_url_base}?frame={frame}", out, "1500,2000", user_data_dir)
            shortcut_frames.append(out)
        make_gif(shortcut_frames, ASSETS / "shortcut.gif",
                  durations_ms=[900, 1400, 2800])
        print(f"wrote {ASSETS / 'shortcut.gif'}")


if __name__ == "__main__":
    main()
