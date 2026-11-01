"""Recompute the size check for an existing catalogue.

    python scripts/recheck_sizes.py               # recompute and write
    python scripts/recheck_sizes.py --dry-run     # show the result without changing anything

New scenes get a verdict during processing. Operator review status is left untouched.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api import db  # noqa: E402
from detector.physical import BORDERLINE, MISMATCH, OK, UNKNOWN, check_size  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--tolerance", type=float, default=0.25, help="allowed deviation from the reference dimensions")
    p.add_argument("--borderline", type=float, default=0.15, help="below this deviation the size counts as matching")
    p.add_argument("--dry-run", action="store_true", help="show the result without writing to the catalogue")
    args = p.parse_args()

    db_path = args.data_dir / "catalog.sqlite"
    if not db_path.exists():
        raise SystemExit(f"catalogue not found: {db_path}")
    db.init(db_path)  # adds the size_check column to older catalogues

    counts = {OK: 0, BORDERLINE: 0, MISMATCH: 0, UNKNOWN: 0}
    updates, flagged = [], []
    with db.session(db_path) as conn:
        rows = conn.execute(
            "SELECT id, scene_id, class_name, confidence, length_m, width_m FROM detections"
        ).fetchall()
        for r in rows:
            check = check_size(r["class_name"], r["length_m"], r["width_m"], args.tolerance, args.borderline)
            counts[check.verdict] += 1
            updates.append((check.verdict, check.deviation, r["id"]))
            if check.verdict in (MISMATCH, BORDERLINE):
                flagged.append((r, check))
        if not args.dry_run:
            conn.executemany(
                "UPDATE detections SET size_check = ?, size_deviation = ? WHERE id = ?", updates
            )

    print(f"detections: {len(rows)}  ·  tolerance {args.tolerance:.0%}")
    print(f"  size matches:        {counts[OK]}")
    print(f"  borderline:          {counts[BORDERLINE]}")
    print(f"  type not confirmed:  {counts[MISMATCH]}")
    print(f"  not checked:         {counts[UNKNOWN]}  (no reference for the class, or no georeferencing)")

    if flagged:
        print("\nneed operator attention:")
        for r, check in sorted(flagged, key=lambda t: -t[1].deviation):
            expected = check.expected_long
            print(f"  scene {r['scene_id']} · id={r['id']:<4} {r['class_name']:<6} conf={r['confidence']:.2f}  "
                  f"measured {r['length_m']:.1f}×{r['width_m']:.1f} m, "
                  f"expected ~{expected[0]:.0f} m along the long axis, {check.deviation:.0%} off")

    if args.dry_run:
        print("\n--dry-run: catalogue unchanged")


if __name__ == "__main__":
    main()
