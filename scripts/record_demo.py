"""Record the README demo GIF (reports/demo.gif) by driving the Streamlit app in a headless browser.

  pip install playwright && python -m playwright install chromium     (once)
  python scripts/record_demo.py                                        (models from .env, < $0.01)
  python scripts/record_demo.py --doc data/samples/transport_layer_tcp_udp_vi.md --out reports/demo_vi.gif

The script starts the app itself, uploads a PUBLIC sample document (never a course document: the GIF is
committed), clicks "Generate questions", waits for the result, opens a question and scrolls to the export
buttons. Long waits are compressed into a few frames.
"""
from __future__ import annotations

import argparse
import io
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url: str, timeout: float = 60) -> None:
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"{url} did not start")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--doc", default="data/samples/transport_layer_tcp_udp.md")
    ap.add_argument("--out", default="reports/demo.gif")
    ap.add_argument("--max-sections", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900, help="seconds to wait for the generation")
    ap.add_argument("--width", type=int, default=1200)
    a = ap.parse_args()

    from playwright.sync_api import sync_playwright

    port = free_port()
    app = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
                            "--server.headless", "true", "--server.port", str(port),
                            "--browser.gatherUsageStats", "false"],
                           cwd=ROOT, env={**os.environ, "PYTHONUTF8": "1"},
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    frames: list[tuple[Image.Image, int]] = []  # (image, milliseconds)
    try:
        wait_http(f"http://127.0.0.1:{port}/_stcore/health")
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": a.width, "height": 760}, device_scale_factor=1)

            def shot(ms: int = 900) -> None:
                frames.append((Image.open(io.BytesIO(page.screenshot())).convert("RGB"), ms))

            page.goto(f"http://127.0.0.1:{port}")
            page.get_by_text("Generate review questions").wait_for(timeout=60_000)
            page.wait_for_timeout(800)
            shot(1500)

            # fewer sections keeps the demo short and cheap
            slider = page.get_by_role("slider", name="Max sections")
            slider.focus()
            for _ in range(20):
                slider.press("ArrowLeft")
            for _ in range(a.max_sections - 1):
                slider.press("ArrowRight")
            page.wait_for_timeout(500)

            page.locator("input[type=file]").set_input_files(str(ROOT / a.doc))
            page.get_by_role("button", name="Generate questions").wait_for(timeout=30_000)
            page.wait_for_timeout(500)
            shot(1500)

            page.get_by_role("button", name="Generate questions").click()
            t0 = time.time()
            done = page.get_by_text("Passed verification")
            while not done.is_visible():
                if time.time() - t0 > a.timeout:
                    raise TimeoutError("generation did not finish: is Ollama running? models in .env?")
                if len(frames) < 6:  # a few frames of the spinner, the rest of the wait is skipped
                    shot(700)
                page.wait_for_timeout(1000)
            page.wait_for_timeout(800)
            shot(2500)

            page.locator("[data-testid='stExpander'] summary").first.click()
            page.wait_for_timeout(800)
            shot(3500)

            page.get_by_text("Export").first.scroll_into_view_if_needed()
            page.wait_for_timeout(600)
            shot(2500)
            browser.close()
    finally:
        app.terminate()
        app.wait(timeout=10)

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    imgs = [im.quantize(colors=128, method=Image.Quantize.FASTOCTREE) for im, _ in frames]
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=[ms for _, ms in frames],
                 loop=0, optimize=True)
    print(f"{out} - {len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB, "
          f"generation took {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
