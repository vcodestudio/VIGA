#!/usr/bin/env python3
"""
Render initialize/*.html to images and evaluate against target PNGs using 5 metrics.

- Renders each HTML via headless Chrome (same flags as servers/generator/html.py)
- Finds target image in data/design2code/testset_final/<id>.png
- Computes metrics from evaluators/design2code/evaluate.py:
  Block, Text, Position, Color (require GT HTML if available), and CLIP (image).
  If GT HTML not found (e.g., Design2Code-HARD/<id>.html missing), those HTML-based
  metrics are omitted for that case.
"""
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any
from PIL import Image

# Reuse metrics
sys.path.append(str(Path(__file__).resolve().parents[2]))  # project root
from evaluators.design2code.evaluate import (
    block_metric,
    text_metric,
    position_metric,
    color_metric,
    clip_similarity,
)


def render_with_chrome(chrome_cmd: str, html_path: str, output_path: str, width: int = 1920, height: int = 1080) -> bool:
    if os.path.exists(output_path):
        return True
    cmd = [
        chrome_cmd,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        f"--window-size={width},{height}",
        f"--screenshot={output_path}",
        f"file://{os.path.abspath(html_path)}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception:
        return False


def evaluate_case(html_path: str, render_path: str, target_png: str, maybe_gt_html: str) -> Dict[str, Any]:
    scores: Dict[str, Any] = {}
    print(f"Evaluating CLIP metric for {render_path} and {target_png}")
    # CLIP metric (image vs image)
    try:
        if os.path.exists(render_path) and os.path.exists(target_png):
            img1 = Image.open(render_path)
            img2 = Image.open(target_png)
            scores["clip"] = float(clip_similarity(img1, img2))
        else:
            scores["clip"] = None
    except Exception as e:
        scores["clip"] = None
        scores["clip_error"] = str(e)

    # HTML-based metrics if GT HTML is available
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            proposed_html = f.read()
        if os.path.exists(maybe_gt_html):
            with open(maybe_gt_html, "r", encoding="utf-8") as f:
                gt_html = f.read()
            scores["block"] = float(block_metric(proposed_html, gt_html))
            scores["text"] = float(text_metric(proposed_html, gt_html))
            scores["position"] = float(position_metric(proposed_html, gt_html))
            scores["color"] = float(color_metric(proposed_html, gt_html))
        else:
            scores.update({"block": None, "text": None, "position": None, "color": None})
    except Exception as e:
        scores.update({"block": None, "text": None, "position": None, "color": None, "html_error": str(e)})

    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate initialize HTMLs against target PNGs with 5 metrics")
    parser.add_argument("--initialize-dir", default="data/design2code/initialize", help="Directory containing generated HTMLs (e.g., <id>.html)")
    parser.add_argument("--targets-dir", default="data/design2code/Design2Code-HARD", help="Directory with target PNGs (<id>.png)")
    parser.add_argument("--gt-html-dir", default="data/design2code/Design2Code-HARD", help="Optional directory with ground-truth HTMLs (<id>.html)")
    parser.add_argument("--chrome-cmd", default="google-chrome", help="Chrome/Chromium command for headless rendering")
    parser.add_argument("--output-dir", default="output/design2code/initialize_eval", help="Directory to save renders and results")
    args = parser.parse_args()

    init_dir = Path(args.initialize_dir)
    tgt_dir = Path(args.targets_dir)
    gt_dir = Path(args.gt_html_dir)
    out_dir = Path(args.output_dir)
    renders_dir = out_dir / "renders"

    out_dir.mkdir(parents=True, exist_ok=True)
    renders_dir.mkdir(parents=True, exist_ok=True)

    html_files = sorted(init_dir.glob("g*.html"), key=lambda p: p.stem)
    if not html_files:
        print(f"No HTML files found in {init_dir}")
        sys.exit(1)

    all_results: Dict[str, Any] = {}
    num = 0
    for html_file in html_files:
        stem = html_file.stem
        target_png = tgt_dir / f"{stem}.png"
        gt_html = gt_dir / f"{stem}.html"
        render_path = renders_dir / f"{stem}.png"

        print(f"Rendering {html_file} -> {render_path}")
        ok = render_with_chrome(args.chrome_cmd, str(html_file), str(render_path))
        if not ok:
            print(f"  Failed to render {html_file}")
            all_results[stem] = {"error": "render_failed"}
            continue

        print(f"Evaluating {stem}")
        scores = evaluate_case(str(html_file), str(render_path), str(target_png), str(gt_html))
        all_results[stem] = {
            "html": str(html_file),
            "render": str(render_path),
            "target_png": str(target_png) if target_png.exists() else None,
            "gt_html": str(gt_html) if gt_html.exists() else None,
            "scores": scores,
        }
        num += 1

    # Aggregate simple averages for available scores
    agg: Dict[str, Any] = {}
    for key in ["block", "text", "position", "color", "clip"]:
        vals = [res["scores"].get(key) for res in all_results.values() if res.get("scores") and isinstance(res["scores"].get(key), (int, float))]
        agg[key] = sum(vals) / len(vals) if vals else None

    summary = {"count": num, "aggregate": agg, "details": all_results}
    with open(out_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {out_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()


