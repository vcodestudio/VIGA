#!/usr/bin/env python3
"""
Generate HTML from PNG mocks using a Vision-Language Model (VLM).

Scans data/design2code/testset_final/*.png and asks the model to reconstruct
the webpage as HTML+CSS. Saves outputs to an output directory as <id>.html.

Environment variables:
- OPENAI_API_KEY (required)
- OPENAI_BASE_URL (optional; defaults to https://api.openai.com/v1)
"""
import os
import sys
import argparse
import base64
import json
from pathlib import Path
from typing import Optional

from openai import OpenAI


def encode_image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def extract_code_block(text: str) -> str:
    """
    Extract HTML/CSS code from a response possibly wrapped in triple backticks.
    Falls back to raw text if no fenced block is detected.
    """
    if not text:
        return ""
    fences = ["```html", "```HTML", "```"]
    start = -1
    for f in fences:
        s = text.find(f)
        if s != -1:
            start = s + len(f)
            break
    if start == -1:
        return text.strip()
    end = text.find("```", start)
    if end == -1:
        return text[start:].strip()
    return text[start:end].strip()


def generate_html_from_image(client: OpenAI, model: str, image_path: str, temperature: float = 0.2) -> str:
    data_url = encode_image_to_data_url(image_path)
    system_prompt = (
        "You are a front-end developer. Given a page screenshot, output a single self-contained "
        "HTML file with embedded CSS (no external assets). Use semantic HTML, flex/grid where appropriate, "
        "and reasonable placeholder text. Keep it valid and renderable."
    )
    user_text = (
        "Reconstruct the page layout and styles from this PNG. "
        "Return ONLY the HTML (with <style> CSS) formatted in a single code block."
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )

    content = resp.choices[0].message.content if resp.choices else ""
    return extract_code_block(content)


def main():
    parser = argparse.ArgumentParser(description="Generate HTML from PNG mocks using a VLM")
    parser.add_argument("--dataset-dir", default="data/design2code/Design2Code-HARD", help="Directory containing *.png mocks")
    parser.add_argument("--output-dir", default="data/design2code/initialize", help="Directory to save generated HTML files")
    parser.add_argument("--model", default="gpt-4o", help="Vision model name")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="API key")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), help="API base URL")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: OPENAI_API_KEY is required (set env var or pass --api-key)")
        sys.exit(1)

    ds = Path(args.dataset_dir)
    if not ds.exists():
        print(f"Error: Dataset directory does not exist: {ds}")
        sys.exit(1)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    png_files = sorted(ds.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not png_files:
        print(f"No PNG files found in {ds}")
        sys.exit(1)

    summary = []
    for png in png_files:
        case_id = png.stem
        try:
            print(f"Processing {case_id}: {png}")
            html = generate_html_from_image(client, args.model, str(png), temperature=args.temperature)
            if not html:
                raise RuntimeError("Empty HTML result")
            out_file = out / f"{case_id}.html"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"Saved: {out_file}")
            summary.append({"case_id": case_id, "status": "success", "output": str(out_file)})
        except Exception as e:
            print(f"Failed {case_id}: {e}")
            summary.append({"case_id": case_id, "status": "error", "error": str(e)})

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Done. Summary saved to {out / 'summary.json'}")


if __name__ == "__main__":
    main()


