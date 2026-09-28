"""Сверка самолётов на снимке с реестром техники.

    python scripts/check_inventory.py data/demo/durban_airport.tif --registry samples/registry_demo.json
    python scripts/check_inventory.py scene.tif --registry reg.json --no-size-check   # для сравнения
    python scripts/check_inventory.py scene.tif --registry reg.json --json

Снимок обрабатывается обеими моделями, затем самолёты сверяются с реестром.
Реестр читается из файла и никуда не сохраняется.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detector.inventory import load_registry, reconcile  # noqa: E402
from detector.registry import ModelRegistry  # noqa: E402
from detector.scene import detect_scene  # noqa: E402

DEFAULT_MODELS = "dota=weights/yolo11s-obb.pt,mar20=weights/mar20_s_800.pt:refine"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", type=Path)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--models", default=os.environ.get("MODELS", DEFAULT_MODELS))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--no-size-check", action="store_true", help="не учитывать сверку габаритов (для сравнения)")
    p.add_argument("--json", action="store_true", help="результат в JSON")
    args = p.parse_args()

    airfields = load_registry(args.registry)
    models = ModelRegistry.from_env(args.models, conf=args.conf)
    known = {c for name in models.refine_only for c in models.class_names[name].values()}
    result = detect_scene(models, args.image)
    report = reconcile(result.features, airfields, gsd_m=result.gsd_m, scene_bounds=result.bounds_wgs84,
                       known_types=known,
                       trust_size_check=not args.no_size_check)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) if args.json else report.to_text())


if __name__ == "__main__":
    main()
