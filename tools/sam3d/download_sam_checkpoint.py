"""Download SAM ViT-H checkpoint to utils/third_party/sam/ (required for /segment, /reconstruct)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "utils" / "third_party" / "sam" / "sam_vit_h_4b8939.pth"
URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
EXPECTED_MB = 2400

def main():
    if OUT.exists() and OUT.stat().st_size > (EXPECTED_MB - 100) * 1024 * 1024:
        print(f"Already exists: {OUT} ({OUT.stat().st_size // 1_000_000} MB)")
        return
    tmp = OUT.with_suffix(".pth.tmp")
    print(f"Downloading to {tmp} (~{EXPECTED_MB} MB) ...")
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
    tmp.replace(OUT)
    print(f"Done: {OUT} ({OUT.stat().st_size // 1_000_000} MB)")


if __name__ == "__main__":
    main()
