"""Download SAM vit_b checkpoint to utils/third_party/sam/ (~375 MB).

Usage (from repo root):
  python tools/sam3d/download_sam_checkpoint.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAM_DIR = ROOT / "utils" / "third_party" / "sam"

FILENAME = "sam_vit_b_01ec64.pth"
URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
EXPECTED_MB = 375


def main():
    out = SAM_DIR / FILENAME
    SAM_DIR.mkdir(parents=True, exist_ok=True)

    if out.exists() and out.stat().st_size > (EXPECTED_MB - 50) * 1024 * 1024:
        print(f"Already exists: {out} ({out.stat().st_size // 1_000_000} MB)")
        return

    tmp = out.with_suffix(".pth.tmp")
    print(f"Downloading vit_b to {tmp} (~{EXPECTED_MB} MB) ...")
    try:
        import requests
        r = requests.get(URL, stream=True, timeout=30)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f:
            for i, chunk in enumerate(r.iter_content(chunk_size=2**20)):
                if chunk:
                    f.write(chunk)
                if total and (i % 50 == 0):
                    print(f"  {f.tell() // 1_000_000} / {total // 1_000_000} MB")
    except Exception as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    tmp.replace(out)
    print(f"Done: {out} ({out.stat().st_size // 1_000_000} MB)")


if __name__ == "__main__":
    main()
