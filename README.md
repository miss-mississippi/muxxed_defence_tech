# Идентификация объектов на космических снимках (NU STeP x Defense Tech Challenge)

Прототип по ТЗ №3 Центра военно-космических программ МО РК: автоматизированное
выявление, распознавание и классификация объектов на космических снимках
с применением ИИ. Модуль детекции + REST API/каталог обнаружений (часть ТЗ №4).

Детектор — YOLO11-OBB (ориентированные боксы): самолёты и корабли стоят под
углом и плотно, горизонтальные боксы их слепляют.

## Быстрый старт

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/infer.py samples/boats.jpg
```

Веса `yolo11s-obb.pt` скачиваются автоматически в `weights/` при первом запуске.
Результат — `outputs/<имя>.json` (детекции) и `outputs/<имя>_obb.jpg` (визуализация).

## Использование из кода

```python
from detector import Detector

det = Detector()                                  # weights/yolo11s-obb.pt, imgsz=1024
dets = det.predict(tile, offset=(x0, y0), rgb=True)  # tile: HxWx3 uint8
for d in dets:
    d.class_name, d.confidence, d.polygon         # polygon — 4 угла в пикселях сцены
```

Формат и конвенция координат — `detector/schema.py`.

## Структура

```
detector/     ML-пакет: загрузка модели, инференс, формат Detection
scripts/      infer.py (инференс), train.py, eval.py
notebooks/    обучение на Kaggle
samples/      тестовые снимки
NOTES.md      решения и метрики
```

## Лицензии и происхождение весов

Мы сознательно используем предобученные веса и трансферное обучение.

- **Ultralytics YOLO** — библиотека, AGPL-3.0, <https://github.com/ultralytics/ultralytics>.
  Поскольку проект использует AGPL-3.0 библиотеку, код проекта также распространяется
  под AGPL-3.0.
- **yolo11s-obb.pt** — предобученные веса Ultralytics, обучены на DOTA-v1.0 (15 классов),
  AGPL-3.0, <https://github.com/ultralytics/assets/releases>.
- **DOTA** — датасет, на котором обучены исходные веса, <https://captain-whu.github.io/DOTA/>.
  По условиям авторов — только для академического использования.
- **samples/boats.jpg** — пример из документации Ultralytics.
