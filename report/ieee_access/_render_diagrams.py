"""Render Mermaid SVGs (with foreignObject text) to PNG via Chromium."""
from pathlib import Path
from playwright.sync_api import sync_playwright

SRC = Path(r"E:\AIDS\Sem 5\23MAT303_Mathematics_for_Intelligent_Systems_5\End Sem Project\Project\docs\diagrams")
DST = Path(__file__).resolve().parent / "figures"

MAPPING = {
    "01_system_architecture.svg": "diag_01_system_architecture.png",
    "02_bayesian_network_from_mermaid.svg": "diag_02_bayesian_network.png",
    "03_train_val_test.svg": "diag_03_train_val_test.png",
    "04_inspection_loop.svg": "diag_04_inspection_loop.png",
    "05_early_warning.svg": "diag_05_early_warning.png",
}

HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  html, body {{ margin: 0; padding: 24px; background: #ffffff; }}
  #wrap {{ display: inline-block; background: #ffffff; }}
  svg {{ background: #ffffff !important; max-width: none !important; height: auto !important; }}
</style></head>
<body><div id="wrap">{svg}</div></body></html>
"""


def main() -> None:
    DST.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 2400, "height": 2400}, device_scale_factor=2)
        for src_name, dst_name in MAPPING.items():
            svg = (SRC / src_name).read_text(encoding="utf-8")
            # Force readable light background for print
            svg = svg.replace("background-color: transparent", "background-color: #ffffff")
            svg = svg.replace('style="max-width:', 'style="background:#ffffff;max-width:')
            page.set_content(HTML.format(svg=svg), wait_until="networkidle")
            page.wait_for_timeout(300)
            box = page.locator("#wrap")
            out = DST / dst_name
            box.screenshot(path=str(out), type="png")
            print(f"OK {dst_name} ({out.stat().st_size} bytes)")
        browser.close()


if __name__ == "__main__":
    main()
