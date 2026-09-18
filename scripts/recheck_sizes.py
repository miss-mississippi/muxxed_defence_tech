"""Сверка габаритов для уже существующего каталога.

Заполняет колонку `size_check` у обнаружений, загруженных до появления физической проверки.
Новые сцены получают вердикт сразу при обработке.

    python scripts/recheck_sizes.py               # пересчитать и записать
    python scripts/recheck_sizes.py --dry-run     # только показать, ничего не менять

Статус проверки оператором (`review_status`) скрипт не трогает: геометрия — подсказка,
решение остаётся за человеком.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api import db  # noqa: E402
from detector.physical import MISMATCH, OK, UNKNOWN, check_size  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--tolerance", type=float, default=0.25, help="допустимое отклонение от паспортных габаритов")
    p.add_argument("--dry-run", action="store_true", help="показать результат, не записывая в каталог")
    args = p.parse_args()

    db_path = args.data_dir / "catalog.sqlite"
    if not db_path.exists():
        raise SystemExit(f"каталог не найден: {db_path}")
    db.init(db_path)  # на старых каталогах добавит колонку size_check

    counts = {OK: 0, MISMATCH: 0, UNKNOWN: 0}
    updates, flagged = [], []
    with db.session(db_path) as conn:
        rows = conn.execute(
            "SELECT id, scene_id, class_name, confidence, length_m, width_m FROM detections"
        ).fetchall()
        for r in rows:
            check = check_size(r["class_name"], r["length_m"], r["width_m"], args.tolerance)
            counts[check.verdict] += 1
            updates.append((check.verdict, check.deviation, r["id"]))
            if check.verdict == MISMATCH:
                flagged.append((r, check))
        if not args.dry_run:
            conn.executemany(
                "UPDATE detections SET size_check = ?, size_deviation = ? WHERE id = ?", updates
            )

    print(f"обнаружений: {len(rows)}  ·  допуск {args.tolerance:.0%}")
    print(f"  габариты сходятся:   {counts[OK]}")
    print(f"  тип не подтверждён:  {counts[MISMATCH]}")
    print(f"  не проверялось:      {counts[UNKNOWN]}  (нет эталона для класса или нет геопривязки)")

    if flagged:
        print("\nтип не подтверждён геометрией:")
        for r, check in sorted(flagged, key=lambda t: -t[1].deviation):
            expected = check.expected_long
            print(f"  сцена {r['scene_id']} · id={r['id']:<4} {r['class_name']:<6} conf={r['confidence']:.2f}  "
                  f"измерено {r['length_m']:.1f}×{r['width_m']:.1f} м, "
                  f"ожидается ~{expected[0]:.0f} м по длинной оси, расхождение {check.deviation:.0%}")

    if args.dry_run:
        print("\n--dry-run: каталог не изменён")


if __name__ == "__main__":
    main()
