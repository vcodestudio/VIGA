#!/usr/bin/env python3
"""
Rollback a static-scene (or similar) run to a given round N.
After rollback, only rounds 0..N-1 are kept; resume will continue from round N.

Round mapping: Round 0 → renders/1, scripts/1.py (1-based on disk).
"""
import argparse
import json
import shutil
from pathlib import Path


def _find_generator_memory_cut_index(memory: list, keep_rounds: int) -> int:
    """Return index at which to slice memory so that exactly `keep_rounds` rounds remain.

    A round = one assistant message with tool_calls + following tool (+ optional user) messages.
    We keep rounds 0..keep_rounds-1, so we cut before the start of round keep_rounds.
    """
    if keep_rounds <= 0:
        return 0
    count = 0
    for i, msg in enumerate(memory):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            count += 1
            if count == keep_rounds + 1:
                return i  # first message of round keep_rounds
    return len(memory)  # keep all


def rollback(output_dir: Path, to_round: int, dry_run: bool) -> None:
    """Rollback run to `to_round`: keep rounds 0..to_round-1 (1-based dirs: 1..to_round)."""
    # Keep rounds 0 .. to_round-1 → keep renders/1..to_round, scripts/1..to_round
    n_keep = to_round
    to_delete_from = n_keep + 1  # 1-based: delete renders/11, scripts/11.py, ...

    if dry_run:
        print(f"[DRY-RUN] Would rollback to round {n_keep} (keep rounds 0..{n_keep - 1}, delete from {to_delete_from} onward)")

    # --- generator_memory.json ---
    gen_path = output_dir / "generator_memory.json"
    if gen_path.exists():
        with open(gen_path, "r", encoding="utf-8") as f:
            memory = json.load(f)
        cut = _find_generator_memory_cut_index(memory, n_keep)
        if dry_run:
            print(f"[DRY-RUN] generator_memory.json: trim to first {cut} messages (of {len(memory)})")
        else:
            with open(gen_path, "w", encoding="utf-8") as f:
                json.dump(memory[:cut], f, indent=4, ensure_ascii=False)
            print(f"Trimmed generator_memory.json to {cut} messages (kept {n_keep} rounds).")

    # --- verifier_memory.json ---
    ver_path = output_dir / "verifier_memory.json"
    if ver_path.exists():
        if dry_run:
            print(f"[DRY-RUN] verifier_memory.json: would clear (empty list)")
        else:
            with open(ver_path, "w", encoding="utf-8") as f:
                json.dump([], f, indent=4, ensure_ascii=False)
            print("Cleared verifier_memory.json.")

    # --- renders/, scripts/, investigator/renders/, investigator/scripts/ ---
    dirs_to_trim = [
        ("renders", True),   # directory per round
        ("scripts", False),  # 1.py, 2.py, ...
        ("investigator/renders", True),
        ("investigator/scripts", False),
    ]
    for subdir, is_dir_per_round in dirs_to_trim:
        base = output_dir / subdir
        if not base.exists():
            continue
        if is_dir_per_round:
            for d in base.iterdir():
                if d.is_dir() and d.name.isdigit() and int(d.name) >= to_delete_from:
                    if dry_run:
                        print(f"[DRY-RUN] delete dir: {d}")
                    else:
                        shutil.rmtree(d)
                        print(f"Removed {d}")
        else:
            for f in base.iterdir():
                if f.suffix == ".py" and f.stem.isdigit() and int(f.stem) >= to_delete_from:
                    if dry_run:
                        print(f"[DRY-RUN] delete file: {f}")
                    else:
                        f.unlink()
                        print(f"Removed {f}")

    # --- blender_file.blend from renders/N/state.blend ---
    state_blend = output_dir / "renders" / str(n_keep) / "state.blend"
    blend_dest = output_dir / "blender_file.blend"
    if state_blend.exists():
        if dry_run:
            print(f"[DRY-RUN] copy {state_blend} -> {blend_dest}")
        else:
            shutil.copy2(state_blend, blend_dest)
            print(f"Restored blender_file.blend from {state_blend}.")
    elif dry_run and not state_blend.exists():
        print(f"[DRY-RUN] no {state_blend}, skip blend restore")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rollback a run to a given round N. Keeps rounds 0..N-1; resume continues from round N."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Task output directory (e.g. output/static_scene/20260204_131221/test2)",
    )
    parser.add_argument(
        "--to-round",
        type=int,
        required=True,
        help="Number of rounds to keep (1-based: keep rounds 0..to-round-1; e.g. 10 keeps renders/1..10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done, do not modify files",
    )
    args = parser.parse_args()

    if not args.output_dir.exists():
        parser.error(f"Output directory does not exist: {args.output_dir}")

    if args.to_round < 1:
        parser.error("--to-round must be >= 1")

    rollback(args.output_dir, args.to_round, args.dry_run)


if __name__ == "__main__":
    main()
