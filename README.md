# Идентификация объектов на космических снимках (NU STeP x Defense Tech Challenge)

Прототип по ТЗ №3 Центра военно-космических программ МО РК: автоматизированное
выявление, распознавание и классификация объектов на космических снимках
с применением ИИ. Плюс минимальный REST API и каталог обнаружений (часть ТЗ №4) —
чтобы показать интеграцию с цифровой платформой ДЗЗ.

Детектор — YOLO11-OBB (ориентированные боксы): самолёты и корабли стоят под
углом и плотно, горизонтальные боксы их слепляют.

## Что умеет

```
GeoTIFF / JPG / PNG
  → тайлы 1024 px с перекрытием, единая растяжка 11/16-бит в 8 бит
  → YOLO11-OBB на каждом тайле (батчами, CUDA / Apple MPS / CPU)
  → склейка дублей на стыках тайлов (IoS, приоритет необрезанным боксам)
  → пиксели → CRS снимка → WGS84: полигон, центр, длина/ширина в метрах, азимут
  → SQLite-каталог → REST API (GeoJSON) → карта Leaflet
  → экспертная проверка: подтвердить / отклонить / исправить класс → отчёт
```

## Быстрый старт

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# один снимок целиком (быстрая проверка)
.venv/bin/python scripts/infer.py samples/boats.jpg

# сцена любого размера → GeoJSON
.venv/bin/python scripts/detect_scene.py path/to/scene.tif

# сервер: карта http://127.0.0.1:8000, документация API http://127.0.0.1:8000/docs
.venv/bin/uvicorn api.app:app --host 127.0.0.1 --port 8000

# тесты
.venv/bin/python -m pytest tests -q
```

Веса `yolo11s-obb.pt` скачиваются автоматически в `weights/` при первом запуске.
Сервер настраивается переменными `WEIGHTS`, `DEVICE` (`cpu`/`mps`/`0`), `CONF`, `DATA_DIR`.

Несколько моделей сразу (общая по DOTA + дообученная по типам техники), порядок задаёт приоритет
при наложении объектов:

```bash
MODELS="dota=weights/yolo11s-obb.pt,mar20=weights/mar20_s_800.pt:refine" .venv/bin/uvicorn api.app:app
```

Суффикс `:refine` — модель только уточняет тип там, где объект нашла общая модель. Без него
модель типов самолётов «узнаёт» их в портовых кранах и контейнерах.

**Режим архивной обработки.** Вместо `mar20_s_800.pt` подключается `mar20_m_800_noise.pt` — та же
модель типов, но архитектуры `m`, обученная с аугментацией гауссовым шумом:

```bash
MODELS="dota=weights/yolo11s-obb.pt,mar20=weights/mar20_m_800_noise.pt:refine" .venv/bin/uvicorn api.app:app
```

Режим покупает +0.023 mAP50 на чистом тесте MAR20 (0.923 против 0.900) ценой +40% времени сцены:
19.0 с против 13.4 с на демо-аэродроме (4096×4096, 25 тайлов, M1/MPS, обе модели). Поэтому по
умолчанию поставляется `s`.

**Режим для зашумлённых снимков.** Устойчивость к сенсорному шуму даёт не размер модели, а
аугментация при обучении: `mar20_s_800_noise.pt` — та же `s`, той же скорости, но держит шум
σ=15 без потерь (0.900) там, где базовая рассыпается до 0.371, а при σ=30 — до 0.063.

```bash
MODELS="dota=weights/yolo11s-obb.pt,mar20=weights/mar20_s_800_noise.pt:refine" .venv/bin/uvicorn api.app:app
```

Три модели типов и когда какая нужна:

| Веса | Когда | Цена |
|---|---|---|
| `mar20_s_800.pt` (по умолчанию) | чистый снимок — штатный случай | на шуме σ=30 падает до 0.063 |
| `mar20_s_800_noise.pt` | снимок зашумлён | на чистом снимке уверенности ниже: при conf ≥ 0.4 показывает 3 объекта там, где базовая 4 |
| `mar20_m_800_noise.pt` | архив, время не критично | +40% времени сцены за +0.023 mAP50 |

Полные таблицы и обоснование — [docs/training/robustness.md](docs/training/robustness.md).

Веса моделей в git не входят (`weights/` в `.gitignore`) и воспроизводятся
`notebooks/kaggle_train.ipynb`: `NOISE=0.5` включает шумовую аугментацию, `MODEL=yolo11m-obb.pt` —
архитектуру `m`.

## REST API

| Метод | Путь | Назначение |
|---|---|---|
| POST | `/api/scenes` | загрузить снимок, обработка в фоне (202) |
| GET | `/api/scenes`, `/api/scenes/{id}` | статус, прогресс, метаданные, счётчики по классам |
| GET | `/api/scenes/{id}/preview.png` | превью сцены (EPSG:3857) для карты |
| GET | `/api/scenes/{id}/report` | отчёт HTML (печать в PDF) |
| GET | `/api/detections` | каталог GeoJSON: `scene_id`, `class_name`, `min_conf`, `status`, `bbox` |
| GET | `/api/detections/{id}` | одно обнаружение |
| PATCH | `/api/detections/{id}` | экспертная проверка `{status, class_name?, comment?}` |
| GET | `/api/stats` | сводка по классам и статусам проверки |
| POST | `/api/detect` | снимок → GeoJSON сразу в ответе, без сохранения (для внешних систем) |
| GET | `/api/health`, `/api/model` | состояние, классы модели |

## Использование из кода

```python
from detector import Detector, detect_scene

det = Detector()                                   # weights/yolo11s-obb.pt, imgsz=1024
dets = det.predict(tile, offset=(x0, y0), rgb=True)  # тайл HxWx3 uint8 → Detection в пикселях сцены
result = detect_scene(det, "scene.tif")            # вся сцена → SceneResult
geojson = result.to_geojson()                      # FeatureCollection в WGS84
```

Формат детекции и конвенция координат — `detector/schema.py`.

## Обучение и оценка

```bash
# MAR20 (военные самолёты, 20 типов) → формат YOLO-OBB
.venv/bin/python scripts/prepare_mar20.py --src /path/to/MAR20 --out data/mar20_yolo

# дообучение с весов DOTA; лучшие веса → weights/<name>.pt, метрики → outputs/eval/
.venv/bin/python scripts/train.py --data data/mar20_yolo/mar20.yaml --name mar20_s_800 --imgsz 800 --batch 16

# оценка: mAP50, mAP50-95, P/R по классам, скорость
.venv/bin/python scripts/eval.py --data data/mar20_yolo/mar20.yaml --weights weights/mar20_s_800.pt --split test
```

Обучение на GPU — `notebooks/kaggle_train.ipynb` (Kaggle, T4/P100):
[открыть в Kaggle](https://www.kaggle.com/kernels/welcome?src=https://github.com/miss-mississippi/muxxed_defence_tech/blob/main/notebooks/kaggle_train.ipynb).

```bash
# проверенные экспертом обнаружения из каталога → датасет YOLO-OBB для дообучения
.venv/bin/python scripts/export_reviewed.py --out data/reviewed_yolo

# производительность на большой сцене: тайлы/с, Мпикс/с, км²/ч
.venv/bin/python scripts/benchmark.py --size 8192 --gsd 0.5
```

## Структура

```
detector/     ML-ядро: модель, тайлинг, склейка, геопривязка, формат Detection
api/          FastAPI + SQLite-каталог, экспертная проверка, отчёт
web/          карта Leaflet (библиотека в web/vendor — работает без CDN)
scripts/      infer, detect_scene, prepare_mar20, train, eval
notebooks/    обучение на Kaggle
tests/        pytest: тайлинг, склейка, геопривязка, конвертер, API
samples/      тестовый снимок
NOTES.md      решения и метрики
```

## Лицензии и происхождение данных

Мы сознательно используем предобученные веса и трансферное обучение.

- **Ultralytics YOLO** — библиотека, AGPL-3.0, <https://github.com/ultralytics/ultralytics>.
  Поскольку проект использует AGPL-3.0 библиотеку, код проекта также распространяется под AGPL-3.0.
- **yolo11s-obb.pt** — предобученные веса Ultralytics, обучены на DOTA-v1.0 (15 классов),
  AGPL-3.0, <https://github.com/ultralytics/assets/releases>.
- **DOTA** — датасет исходных весов, <https://captain-whu.github.io/DOTA/>. По условиям авторов —
  только для академического использования.
- **MAR20** — датасет для дообучения (военные самолёты), CC BY-NC 4.0, <https://gcheng-nwpu.github.io/>.
- **Демо-снимки** — Maxar Open Data Program, CC BY-NC 4.0, © Maxar Technologies,
  <https://registry.opendata.aws/maxar-open-data/>. Не хранятся в репозитории, скачиваются
  `scripts/fetch_demo_scene.py`.
- **Leaflet 1.9.4** — BSD-2-Clause, <https://leafletjs.com>.
- Подложки карты: Esri World Imagery и OpenStreetMap — только онлайн, с атрибуцией.
- **samples/boats.jpg** — пример из документации Ultralytics.
