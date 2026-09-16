"""Captures a handful of deterministic states from the two interactive explorer pages
via headless Chrome screenshots, then stitches each set into a looping GIF. Uses the
`?reveal=1`/`?example=N` and `?frame=N` query-param hooks those pages expose
specifically for this (see the bottom of each page's <script>), not real click/wait
automation -- there is no browser-automation dependency, only a local Chrome install.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXPLORER = ROOT / "explorer"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

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
            f"--user-data-dir={user_data_dir}",
            f"--screenshot={out_path}", f"--window-size={window}", url,
        ],
        check=True, capture_output=True, timeout=30,
    )


def make_gif(frame_paths: list[Path], out_path: Path, durations_ms: list[int]) -> None:
    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:], duration=durations_ms, loop=0,
    )


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
                       out, "1000,1150", user_data_dir)
            room_frames.append(out)
        make_gif(room_frames, ASSETS / "interrogation_room.gif",
                  durations_ms=[1400, 2600, 900])
        print(f"wrote {ASSETS / 'interrogation_room.gif'}")

        shortcut_url_base = (EXPLORER / "shortcut.html").as_uri()
        shortcut_frames = []
        for i, frame in enumerate((0, 1, 2)):
            out = tmp_path / f"shortcut_{i}.png"
            screenshot(chrome, f"{shortcut_url_base}?frame={frame}", out, "1100,800", user_data_dir)
            shortcut_frames.append(out)
        make_gif(shortcut_frames, ASSETS / "shortcut.gif",
                  durations_ms=[900, 1400, 2800])
        print(f"wrote {ASSETS / 'shortcut.gif'}")


if __name__ == "__main__":
    main()
