# Engineering notes: decisions and measurements

A dated log of what we decided, what we measured, and what turned out to be wrong. Entries are in
chronological order (September 2026). Negative results and corrections are kept on purpose.

## Timeline

- Sep 15: kick-off, working inference with a pretrained model
- Sep 16: fine-tuning on Kaggle
- Sep 17: feature freeze
- Sep 19–20: solution and presentation submitted
- Sep 29: Demo Day (finalist)

## Sep 15: Setup and architecture

### Model: YOLO11-OBB pretrained on DOTA-v1.0
- Oriented boxes: aircraft and ships sit at arbitrary angles and close together; horizontal boxes
  merge neighbouring objects.
- Start from `yolo11s-obb` as a speed/accuracy trade-off. Transfer learning instead of training
  from scratch is a deliberate choice: four days and very little labelled regional data.
- Model input is 1024 px, the size the DOTA weights were trained at.

### Hardware
- Local: MacBook with Apple M1, 8 GB, for development and inference.
- Training: Kaggle (T4/P100, ~30 GPU-hours per week, background runs). Training OBB at 1024 px on
  8 GB of unified memory is not realistic in time or memory.

### Contract with the geo service
- The ML module returns detections in **whole-scene pixels**: 4 OBB corners, centre/size/angle,
  class, confidence (`detector/schema.py`).
- Pixels -> image CRS through the rasterio affine transform, then -> WGS84.
- Deduplication on tile overlaps is the ML module's job.
- We decided to build the whole minimal end-to-end chain ourselves (tiling, georeferencing,
  GeoJSON, API, map) and keep the interfaces documented, so any piece can be replaced.

### Tiling and merging (`detector/scene.py`, `detector/merge.py`)
- 1024 px tiles, 200 px overlap. The last row/column is snapped to the scene edge, so every tile
  is full-size and the model never resizes it.
- Non-uint8 imagery (11/16-bit) is stretched to uint8 at the 2–98 percentiles with **one pair of
  limits for the whole scene** (from a downsampled copy); otherwise contrast jumps between tiles.
  uint8 imagery is left as is.
- R, G, B bands are chosen from the GeoTIFF colour interpretation, otherwise the first three;
  panchromatic is replicated to 3 channels.
- Empty tiles (<5% valid pixels) are skipped. Validity comes from the GeoTIFF mask and the **raw**
  values. Bug found on Sep 15: validity was first computed after stretching, where dark water
  becomes 0, so an open-sea tile with ships was thrown away as empty. The same bug punched holes
  in the preview. Fixed and covered by a test.
- Merging: IoS (intersection / smaller polygon) >= 0.6, only between different tiles and only for
  the same class. IoU does not work here: at a tile edge the object is truncated, and the IoU between
  the full and the truncated box is small. Detections that do not touch an inner tile edge win,
  then the more confident ones.
- Georeferencing: pixel -> image CRS (affine) -> WGS84 (pyproj). The GeoJSON carries the polygon,
  the centre, geodesic length/width in metres, the long-axis azimuth (0–180°) and the scene GSD.
  A warning is raised if the GSD is coarser than 2 m/px.

## Sep 15: Baseline measurements

### Pretrained model inference (no fine-tuning)

`yolo11s-obb.pt`, ultralytics 8.4.152, torch 2.14.0, conf=0.25, median of 5 runs after warm-up.

| Image | Size | Device | imgsz | Detections | End-to-end, ms |
|---|---|---|---|---|---|
| samples/boats.jpg | 1920×1080 | M1 MPS | 1024 | 174 (ship 171, harbor 3) | 95.8 |
| samples/boats.jpg | 1920×1080 | M1 CPU | 1024 | 174 (ship 171, harbor 3) | 228.6 |

- MPS and CPU detections are identical (centres within 0.01 px, same classes and confidences), so
  MPS is correct for OBB inference.
- The model time reported by MPS in `speed["inference"]` (13 ms) is too low because MPS is
  asynchronous, so we time end-to-end.
- boats.jpg is a drone photo, not a satellite image: it checks the pipeline, not quality on our task.
- A 1920×1080 frame is shrunk to 1024 on its long side; on real scenes tiles are ~1024 px with no
  downscaling.

### Tiled scene (synthetic GeoTIFF: boats.jpg -> uint16, UTM 42N, 0.1 m/px)

| Mode | Tiles | Detections | Scene time, s (M1 MPS) |
|---|---|---|---|
| no merging | 6 | 588 (ship 583, harbor 5) | 3.3 |
| IoS 0.6 merging | 6 | 198 (ship 195, harbor 3) | 4.2 |

- Full-resolution tiles found more small boats than the whole downscaled frame (198 vs 174).
- No duplicates visible on the overlaps.

### Throughput on a large scene (`scripts/benchmark.py`)

Synthetic 8192×8192 scene at 0.5 m/px (16.8 km²), 1024 tiles / 200 overlap, batch 4, yolo11s-obb.

| Device | Tiles | Scene time, s | Tiles/s | Mpx/s | km²/h |
|---|---|---|---|---|---|
| M1 MPS | 100 | 16.8 | 5.95 | 3.99 | ~3600 |
| M1 CPU | 100 | 57.4 | 1.74 | 1.17 | ~1050 |

- The time covers everything: GeoTIFF reading, stretching, inference, merging, georeferencing
  (5988 objects).
- km²/h scales with the square of the GSD: at 0.3 m/px the same ~6 tiles/s give ~1300 km²/h.

### Training and evaluation smoke tests (dota8: 4+4 images shipped with ultralytics)

The goal is to check that `scripts/train.py` and `scripts/eval.py` work, **not** quality: dota8 is
taken from DOTA train, so the pretrained model has seen these images.

| Run | Device | imgsz | mAP50 | mAP50-95 | Inference, ms/image |
|---|---|---|---|---|---|
| yolo11s-obb (no fine-tuning), val | M1 MPS | 1024 | 0.995 | 0.855 | 394* |
| smoke_dota8, 1 epoch, val | M1 CPU | 640 | 0.995 | 0.871 | 179 |

\* on MPS the ultralytics timing includes warm-up and asynchrony; not a speed metric.

## Data: MAR20 for fine-tuning

- MAR20: 3842 Google Earth images of ~800×800, 60 military airfields, 20 military aircraft types
  (SU-35, C-130, C-17, C-5, F-16, TU-160, E-3, B-52, P-3C, B-1B, E-8, TU-22, F-15, KC-135, F-22, FA-18,
  TU-95, KC-10, SU-34, SU-24), 22341 objects, OBB labels. Official split: 1331 train / 2511 test.
- Licence CC BY-NC 4.0: non-commercial use, fine for a hackathon, stated in the README.
- Role in the system: DOTA gives *detection* (plane/ship/vehicle…), MAR20 gives *recognition and
  classification* of military aircraft types.
- Format: XML with `robndbox` (4 corners) -> `scripts/prepare_mar20.py` -> YOLO-OBB.
- In the Kaggle copy of MAR20 (`khlaifiabilel/military-aircraft-recognition-dataset`, 1.2 GB,
  original layout), **4 XML files out of 3842** have a zero `size`, and the converter crashed with
  a division by zero while normalising coordinates. The size is now read from the JPEG itself when
  the XML lacks it. Covered by a test.
- Conversion verified on Kaggle: 3842 images (train 1131 / val 200 / test 2511), 22341 objects,
  matching the MAR20 description, no missing xml/jpg pairs.
- Images are not strictly 800×800 (e.g. 859×831 occurs); we train at imgsz 800 and ultralytics
  letterboxes them.
- The Hugging Face mirror (Alex5666/Military-Aircraft-Recognition-dataset) is **incomplete: 100 of
  3842 images** (checked through the API), so it is useless for training.
- The full archive is on the official OneDrive/Baidu links at gcheng-nwpu.github.io; on Kaggle it is
  uploaded as a private dataset (the notebook finds it in /kaggle/input and fails if there are
  fewer than 3000 images).

## Demo scenes: real satellite imagery (Maxar Open Data)

- Source: Maxar Open Data Program, CC BY-NC 4.0 (© Maxar Technologies), event
  `southafrica-flooding22`, WorldView capture on 2022-04-20, GSD 0.38–0.39 m, ~28° off-nadir,
  0% cloud. ARD tiles 17408×17408, 0.305 m/px, UTM 36S (EPSG:32736), uint8 RGB.
- Why Durban: the largest container port in Africa plus King Shaka airport, at a resolution close
  to DOTA. Florida (Hurricane Ian) is 0.6–0.9 m, too coarse; Maui does not cover Kahului airport.
- 4096×4096 windows (~1.25 km) are cut from the COG via HTTP range requests
  (`scripts/fetch_demo_scene.py`) and not committed: the port terminal, an oil terminal with
  storage tanks, and the airport.

### Pretrained yolo11s-obb on the real scenes (no fine-tuning)

M1 MPS, 1024 tiles / 200 overlap, 25 tiles per scene. **Visual assessment without ground truth;
this is not mAP.**

| Scene | conf ≥ 0.25 | conf ≥ 0.4 | Time, s |
|---|---|---|---|
| King Shaka airport | 8: plane 7, soccer ball field 1 | 5: plane 5 | 5.9 |
| Container terminal | 40: large vehicle 31, ship 8, roundabout 1 | 22: large vehicle 17, ship 4, roundabout 1 | 8.5 |
| Oil terminal | 154: storage tank 114, large vehicle 15, swimming pool 10, small vehicle 9, harbor 3, ship 2, ground track field 1 | 129: storage tank 105, large vehicle 11, swimming pool 9, ship 2, harbor 1, ground track field 1 | 7.0 |

- Airport: 5 aircraft on the apron, all 5 found at conf ≥ 0.4 with no false positives. At 0.25,
  two truncated fragments of the same aircraft appear (0.33–0.38, elongated 33×15 m), plus a
  "soccer ball field" on a concrete pad (0.26).
- Terminal: 2 container ships are boxed whole (0.54–0.64); the third, under the cranes, breaks into
  low-confidence pieces (0.27–0.34). Containers are not detected: DOTA has no such class. Trucks on
  the roads come out as large vehicle.
- Oil terminal: nearly all storage tanks (0.9+), both tankers at the berth, plausible swimming
  pools in the residential block.
- The leftovers are not nested duplicates: pairs of the same class with IoS ≥ 0.6 are almost absent
  (1 on the terminal). Merging cannot be made more aggressive without fusing neighbouring objects.
  This is handled by the confidence threshold and by fine-tuning.
- **Decision:** the catalogue stores everything at conf ≥ 0.25 (the operator sees doubtful
  candidates); the map and report show conf ≥ 0.4 by default.

## Sep 15: Operator review -> fine-tuning data (`scripts/export_reviewed.py`)

- Map: confirm (C) / reject (R) / correct the class; after a decision, focus moves to the next object.
- Export: only tiles in which every intersecting detection has been reviewed go into the dataset
  (confirmed -> labels, rejected -> background). One unreviewed object in a tile would turn it into
  "background" and teach the model to miss objects.
- Validation uses separate scenes: tiles of one scene overlap, so a random tile split would leak.
- Limitation: model misses cannot be labelled this way; that needs a tool like CVAT (the YOLO-OBB
  format is compatible).

## Sep 16: Fine-tuning on MAR20

`yolo11s-obb.pt` (DOTA) -> 60 epochs, imgsz 800, batch 16, DDP on two T4s. The whole run took
25 minutes (1511 s), ~20 s per epoch. Metrics are on the **official MAR20 test set** (2511 images,
14471 objects); val was carved out of train, so epoch selection does not leak.

**mAP50 0.899 · mAP50-95 0.786 · P 0.879 · R 0.841 · inference 15.7 ms/image (T4)**

| Class | Images | Instances | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| SU-35 | 233 | 1092 | 0.893 | 0.881 | 0.920 | 0.802 |
| C-130 | 299 | 993 | 0.995 | 0.851 | 0.975 | 0.867 |
| C-17 | 230 | 860 | 0.885 | 0.984 | 0.972 | 0.865 |
| C-5 | 103 | 463 | 0.975 | 0.860 | 0.956 | 0.869 |
| F-16 | 153 | 827 | 0.586 | 0.874 | 0.747 | 0.631 |
| TU-160 | 79 | 334 | 0.997 | 0.940 | 0.988 | 0.873 |
| E-3 | 119 | 489 | 0.937 | 0.971 | 0.985 | 0.893 |
| B-52 | 126 | 579 | 0.906 | 0.901 | 0.923 | 0.823 |
| P-3C | 167 | 611 | 0.919 | 0.936 | 0.954 | 0.853 |
| B-1B | 116 | 556 | 0.937 | 0.975 | 0.990 | 0.880 |
| E-8 | 125 | 356 | 0.915 | 0.818 | 0.915 | 0.820 |
| TU-22 | 161 | 462 | 0.883 | 0.897 | 0.937 | 0.779 |
| F-15 | 195 | 1091 | 0.712 | 0.716 | 0.745 | 0.633 |
| KC-135 | 304 | 1119 | 0.939 | 0.844 | 0.948 | 0.831 |
| F-22 | 107 | 418 | 0.524 | 0.656 | 0.589 | 0.491 |
| FA-18 | 216 | 1847 | 0.983 | 0.443 | 0.824 | 0.712 |
| TU-95 | 277 | 958 | 0.976 | 0.941 | 0.978 | 0.852 |
| KC-10 | 62 | 148 | 0.845 | 0.689 | 0.846 | 0.769 |
| SU-34 | 219 | 717 | 0.835 | 0.809 | 0.870 | 0.737 |
| SU-24 | 116 | 551 | 0.944 | 0.829 | 0.910 | 0.739 |

- The hard classes are fighters: F-22 (mAP50 0.589), F-15 (0.745), F-16 (0.747). They are small and
  look alike. FA-18 has precision 0.983 but recall 0.443: the model misses them rather than
  confusing them. This is the expected failure of small-object fine-grained classification with
  20 classes and 1131 training images.
- Large, distinctive silhouettes (B-1B, TU-160, E-3, TU-95, C-130) reach 0.97–0.99.
- Report the per-class breakdown, not only the mean.

## Sep 16: Two models at once (`detector/registry.py`)

- The MAR20 model knows only 20 aircraft types. If we simply swapped the weights, ships, storage
  tanks and vehicles would disappear from the map. So the models run together.
- Every detection carries its model name (`model` in GeoJSON, a catalogue column, an API filter).
- One object found by both models (DOTA "plane" and MAR20 "SU-34") collapses into one detection:
  the higher-priority model wins, i.e. the specialised one. Classes are not compared because the
  models have different class sets.
- Launch: `MODELS="dota=weights/yolo11s-obb.pt,mar20=weights/mar20_s_800.pt" uvicorn api.app:app`;
  the order sets the priority.
- Cost: scene time is the sum of the models, two passes over the tiles instead of one.

## Sep 16: The specialised model only refines (`:refine`)

Running MAR20 together with DOTA on the real Durban scenes exposed two failures:

- **Phantom aircraft away from the airfield.** On the container terminal MAR20 produced four
  "TU-95" at 0.71 / 0.62 / 0.36 / 0.29: cranes and container stacks. A threshold cannot fix this:
  two of those confidences are higher than DOTA's confidence on real ships.
- **Two types on one object.** A single aircraft got C-17 (0.77) and C-5 (0.55) at once: the boxes
  came from neighbouring tiles, and the cross-tile merge compared classes.

Fixes:
- `ModelSpec(refine_only=True)`: the refining model's detections are kept only where the general
  model found the same object. After this MAR20 produces nothing on the terminal or the oil
  terminal, and the types stay on the airfield.
- `agnostic_nms=True` inside the model and class-agnostic cross-tile merging for such models: an
  object cannot be two types at once.
- In `MODELS` this is enabled with the `:refine` suffix.

End-to-end result through the API on the demo scenes (conf ≥ 0.4, both models, 25 tiles per scene):

| Scene | Detections |
|---|---|
| King Shaka airfield | mar20: C-17 ×2, E-8, C-5 + dota: plane ×1 (type not determined) |
| Container terminal | dota: large vehicle 17, ship 4, roundabout 1; nothing from mar20 |
| Oil terminal | dota: storage tank 105, large vehicle 11, swimming pool 9, ship 2, harbor 1, ground track field 1; nothing from mar20 |

- A 4096×4096 scene with two models takes about 10 s on the M1 (one pass was ~6–8 s).
- When merging duplicates, the box not truncated by a tile edge wins over the more confident one:
  on one aircraft the C-5 label (0.55) survived although the truncated box from the neighbouring
  tile said C-17 (0.77). The rule is deliberate, and we state it openly.

**A limitation to state out loud:** MAR20 has no civilian classes, so a civilian airliner gets the
nearest military type (C-17, C-5, E-8 in Durban) with high confidence. The model is meant for
military airfields; on civilian ones its type output is unreliable, and the operator can see this
from the model tag on every detection.

## Sep 16: Kazakhstan imagery: no open source at the required resolution

- Maxar Open Data: no events in Kazakhstan at all; the programme covers natural disasters.
- OpenAerialMap (CC BY 4.0): 35 captures in the 46–87°E, 40–55°N box, exactly one of them in
  Kazakhstan, a mountain trail near Almaty (1.4 cm/px, 0.4 km). The rest are Ulyanovsk, Ufa and
  Kyrgyzstan. No airfields or ports.
- Sentinel-2 (10 m) is free but useless for aircraft and ships: an object is 2–5 pixels.
- Esri/Google basemaps may be displayed in the app with attribution, but downloading them as input
  imagery for detection violates their terms of use, so we do not do it.

**Conclusion:** imagery of Kazakhstan has to come from the organisers or the customer
(KazEOSat-1/2). The pipeline does not depend on the source: any georeferenced GeoTIFF is processed
as is. The demo stays on Durban, and we say so explicitly.

## Sep 16: Robustness to image quality (`scripts/robustness.py`)

`mar20_s_800.pt`, 400 MAR20 test images, imgsz 800, T4. Full table:
[docs/training/robustness.md](docs/training/robustness.md).

| Degradation | mAP50 | Change |
|---|---|---|
| original | 0.900 | — |
| noise σ=15 | 0.371 | −58.8% |
| noise σ=30 | 0.063 | −93.0% |
| blur 5 / 9 px | 0.901 / 0.893 | +0.1% / −0.8% |
| JPEG q=30 | 0.866 | −3.8% |
| resolution ÷2 / ÷4 | 0.901 / 0.857 | +0.1% / −4.8% |
| brightness ×0.5 | 0.889 | −1.2% |
| haze 30% | 0.901 | +0.1% |

- The model handles blur, compression, halved resolution, low light and haze well.
- **The only real weakness is sensor noise.** The training augmentations had no noise. Next step:
  add Gaussian noise to the augmentations and re-measure.
- The 0.900 baseline on 400 images matches 0.899 on the full 2511-image test set, so the subset is
  representative.
- A repeated training run gave exactly the same metrics (0.899 / 0.786): reproducibility confirmed.

## Sep 16: Operator review of the demo scenes: precision on real imagery

All 201 detections on the three Durban scenes were reviewed (catalogue threshold 0.25), with no
class corrections at this stage.

| Scene | Confirmed | Rejected | Precision |
|---|---|---|---|
| King Shaka airfield | 5 | 2 | 0.714 |
| Container terminal | 34 | 6 | 0.850 |
| Oil terminal | 150 | 4 | 0.974 |
| **Total** | **189** | **12** | **0.940** |

Per class: storage tank 0.974 (111/3), large vehicle 0.978 (45/1), swimming pool 0.900 (9/1);
small vehicle, harbor, roundabout, ground track field 1.000; **ship 0.500 (5/5)**, plane 0.500 (1/1),
soccer ball field 0.000 (0/1). All aircraft types (C-17 ×2, E-8, C-5) were confirmed.

- **The main weakness shows up in numbers:** ships at 0.500 are the fragments of the container ship
  under the cranes that we had already seen. Fine-tuning on port scenes would address it.
- **This is precision, not recall.** Model misses do not enter the review: the operator sees only
  what the model found. Recall on these scenes is unknown, and we say so.
- **Confirming the object is not confirming the type.** The reviewer answered "the object is
  there", but Durban is civilian, so military types there are nominal. (See the Sep 18 review.)

## Sep 16: Closing the loop: review -> dataset (`scripts/export_reviewed.py`)

| Dataset | Tiles | Labels | Foreign classes |
|---|---|---|---|
| for the general model (DOTA classes) | 51 (28 train / 23 val) | 331 | 5 (aircraft types) |
| for the type model (MAR20 classes) | 51 (28 / 23) | 5 | 331 (general classes) |

- There were no unreviewed tiles, so all of them went in.
- There are more labels than objects: an object on a tile overlap goes into both tiles, which is
  fine for training (the model sees it in two contexts).
- Each model is fine-tuned on its own class list; "foreign" objects do not enter its dataset. Nothing
  is lost: for the general model an aircraft stays `plane`, for the type model it is `C-17`.

## Sep 16: Model comparison: yolo11s vs yolo11m

Same conditions: 60 epochs, imgsz 800, batch 16, 2×T4, seed 0, MAR20 test (2511 images).

| Model | mAP50 | mAP50-95 | P | R | Inference, ms/image (T4) |
|---|---|---|---|---|---|
| yolo11s-obb | 0.899 | 0.786 | 0.879 | 0.841 | **15.7** |
| yolo11m-obb | **0.904** | **0.795** | **0.885** | **0.858** | 43.7 |
| difference | +0.005 | +0.009 | +0.006 | +0.017 | 2.8× slower |

The mean barely moved, but the models differ noticeably per class:

| Class | s | m | Δ |
|---|---|---|---|
| F-22 | 0.589 | 0.717 | **+0.128** |
| KC-10 | 0.846 | 0.861 | +0.015 |
| F-15 | 0.745 | 0.754 | +0.009 |
| C-130 | 0.975 | 0.972 | −0.003 |
| SU-35 | 0.920 | 0.911 | −0.009 |
| FA-18 | 0.824 | 0.804 | −0.020 |
| F-16 | 0.747 | 0.649 | **−0.098** |

**Decision: yolo11s stays the default.** The +0.5% in mean mAP50 is within run-to-run variance, while
speed drops 2.8×. For a system that scans thousands of km², throughput matters more than the third
decimal place.

- m "fixes" F-22 (+0.128) but drops F-16 (−0.098): both models struggle with small look-alike
  fighters, just differently. This is a data problem, not a capacity problem: these types need more
  examples.
- m makes sense as an archive-processing mode where time is not critical; the model registry switches
  it with one environment variable.
- *Update, Sep 18:* the weights of an `m` trained with noise augmentation were downloaded
  (`weights/mar20_m_800_noise.pt`), and the archive mode is enabled through `MODELS` (see Sep 18).

## Sep 16: Operational lesson: long runs only via Save & Run All

The noise-augmentation experiment failed twice for reasons unrelated to code:

- the interactive Kaggle session resets on its own, and every reset wipes `/kaggle/working`: the repo
  clone, the built dataset and running processes are gone;
- processes started from the session console (`!nohup`, `subprocess.Popen`) do not survive a reset;
- attached input datasets (`/kaggle/input`) do survive.

**Rule: anything longer than a couple of minutes runs through Save Version -> Save & Run All.** It is a
server-side batch that does not depend on the browser tab or the session. The first successful training
run worked exactly this way. The session console is only for quick checks.

Nothing important was lost: all metrics (`s`, `m`, robustness) were recorded here right after they were
produced, and the `mar20_s_800.pt` weights were downloaded. A `NOISE` parameter was added to the
notebook, so the noise run starts as a normal batch without editing cells.

## Sep 17: Training with noise augmentation: clean accuracy unaffected

Run `mar20_s_800_noise`, Save & Run All batch, 29 minutes (1755 s), 2×T4. The log confirms
`noise: 0.5` and `GaussNoise(p=0.5)` in the albumentations pipeline.

| Model | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|
| yolo11s without noise | 0.899 | 0.786 | 0.879 | 0.841 |
| yolo11s + GaussNoise(0.5) | 0.898 | 0.784 | 0.857 | 0.851 |

- **Noise augmentation did not cost clean accuracy**: a 0.001 mAP50 difference is measurement noise.
  Precision dropped by 0.022 and recall rose by 0.010: the model became slightly bolder.
- The robustness step did not run in this batch (no `robustness.md` / `robustness.json` in the output),
  so the effect on noisy images was not measured yet. *Closed on Sep 18, see below.*

## Sep 18: Fourth run: yolo11m with noise augmentation, best accuracy

The run was launched as `mar20_s_800_noise2`, but the log clearly shows `YOLO11m-obb summary:
20,917,263 parameters` and a download of `yolo11m-obb.pt`. **An m model was trained; the run name is
misleading**: `RUN_NAME` only sets the folder, `MODEL` sets the architecture. 40-minute batch, 2×T4,
`GaussNoise(p=0.5)` confirmed in the pipeline.

| Run | Architecture | Noise | mAP50 | mAP50-95 | P | R | Inference |
|---|---|---|---|---|---|---|---|
| mar20_s_800 | s | no | 0.899 | 0.786 | 0.879 | 0.841 | **15.7 ms** |
| mar20_s_800_noise | s | yes | 0.898 | 0.784 | 0.857 | 0.851 | 18.3 ms |
| mar20_m_800 | m | no | 0.904 | 0.795 | 0.885 | 0.858 | 43.7 ms |
| mar20_s_800_noise2 | **m** | **yes** | **0.915** | **0.804** | 0.895 | 0.880 | 40.4 ms |

- **Noise helped the large model and did not hurt the small one.** m gains +0.011 mAP50 and +0.022
  recall; for s the changes are within run-to-run variance (±0.005).
- The best accuracy of all runs is m with noise: 0.915, at the same cost, 2.6× slower than s.
- **The shipping decision does not change: s without noise stays the default.** For a system that
  scans thousands of km², 15.7 ms vs 40.4 ms matters more than 1.6% mAP50.
- The robustness step was again missing from the output: Kaggle's import-by-link pulled a stale
  version of the notebook from GitHub. Fix: the measurement is now built into `scripts/train.py`
  (runs after evaluation, disabled with `--robustness 0`), and a separate notebook,
  `notebooks/kaggle_robustness.ipynb`, measures robustness on trained weights in 10 minutes.

### Per class: m with noise vs m without noise (mAP50)

| Class | m + noise | Δ vs m |
|---|---|---|
| B-1B | 0.995 | — |
| E-3 | 0.994 | — |
| B-52 | 0.993 | — |
| TU-160 | 0.992 | — |
| TU-95 | 0.985 | — |
| P-3C | 0.970 | — |
| KC-135 | 0.962 | — |
| C-17 | 0.962 | — |
| C-130 | 0.951 | −0.021 |
| TU-22 | 0.951 | — |
| C-5 | 0.936 | — |
| SU-35 | 0.913 | +0.002 |
| FA-18 | 0.907 | **+0.103** (recall 0.444 -> 0.737) |
| SU-24 | 0.902 | — |
| SU-34 | 0.895 | — |
| E-8 | 0.874 | — |
| F-15 | 0.839 | **+0.085** |
| KC-10 | 0.836 | −0.025 |
| F-16 | 0.776 | **+0.127** |
| F-22 | 0.672 | −0.045 |

- Noise lifted exactly the classes that were failing: F-16, F-15 and especially FA-18, whose recall
  rose from 0.444 to 0.737; the model stopped missing these aircraft.
- F-22 dropped by 0.045 and remains the worst class: 418 objects on 107 images, a small low-observable
  aircraft. That is a data question, not an augmentation question.

## Sep 18: The noise weakness is closed: measured

Measured on trained weights with `notebooks/kaggle_robustness.ipynb` (490 s, no retraining),
model `mar20_s_800_noise2.pt`: architecture m, trained with `GaussNoise(p=0.5)`.

| Degradation | before: s without noise | after: m with noise |
|---|---|---|
| original | 0.900 | 0.923 |
| **noise σ=15** | 0.371 (−58.8%) | **0.918 (−0.5%)** |
| **noise σ=30** | 0.063 (−93.0%) | **0.906 (−1.9%)** |
| blur 9 px | 0.893 (−0.8%) | 0.901 (−2.4%) |
| JPEG q=30 | 0.866 (−3.8%) | 0.917 (−0.7%) |
| resolution ÷4 | 0.857 (−4.8%) | 0.883 (−4.4%) |
| brightness ×0.5 | 0.889 (−1.2%) | 0.927 (+0.4%) |
| haze 30% | 0.901 (+0.1%) | 0.911 (−1.3%) |

- **The full loop is closed on measurements:** found the weakness by measuring -> formed a hypothesis
  (no noise in augmentations) -> added the augmentation -> retrained -> re-measured with the same
  protocol. The loss under noise fell from 93% to 2%.
- The measurement notebook worked the first time and picked up weights from another notebook's
  output: "measure on trained weights without retraining" works, 10 minutes instead of 40.
- **Caveat:** two factors changed between the tables, architecture and augmentation. Separating them
  needs the same measurement on s trained with noise. *Done the same day, see below.*

## Sep 18: Archive mode wired in: `m` with noise via `MODELS`

The weights were taken from the Kaggle output into `weights/mar20_m_800_noise.pt`. On Kaggle the file
is called `mar20_s_800_noise2.pt` (the run name is left over from `s`), so it was renamed locally;
loading it confirms the architecture: **20.9M parameters, 20 MAR20 classes**.

Checked on the demo airfield (4096×4096, 25 tiles, M1/MPS, both models, conf 0.25):

| Configuration | Scene time | What MAR20 found |
|---|---|---|
| `s` (default) | 13.4 s | C-17 0.83, C-17 0.67, E-8 0.83, C-5 0.55 |
| `m` + noise (archive) | 19.0 s | E-8 0.88, E-3 0.74, E-8 0.69, C-17 0.43 |

- No code changes were needed: switching is one variable,
  `MODELS="dota=weights/yolo11s-obb.pt,mar20=weights/mar20_m_800_noise.pt:refine"`.
  The model registry was written exactly for this.
- Both find the same number of objects (4), but **the types differ**. That does not make one better:
  King Shaka is civilian and MAR20 has no civilian classes, so both models pick the nearest military
  type. Type correctness cannot be judged on this scene; on the clean MAR20 test `m` with noise scores
  0.923 vs 0.900 for `s`.
- The cost of the mode is **+40% scene time**, so `s` stays the default and `m` is used for archive
  and known-noisy imagery.
- Weights are not committed (`weights/` is in `.gitignore`, 42 MB); they are reproduced with
  `notebooks/kaggle_train.ipynb` using `MODEL=yolo11m-obb.pt` and `NOISE=0.5`.

## Sep 18: Settled: augmentation fixed the collapse, not model size

Between "before" and "after" two factors changed at once: architecture (`s` -> `m`) and augmentation
(none -> noise). We ran the same protocol on `s` trained with noise (`mar20_s_800_noise.pt`):

| Degradation | s without noise | **s with noise** | m with noise |
|---|---|---|---|
| original | 0.900 | 0.900 | 0.923 |
| noise σ=15 | 0.371 (−58.8%) | **0.900 (−0.1%)** | 0.918 (−0.5%) |
| noise σ=30 | 0.063 (−93.0%) | **0.890 (−1.2%)** | 0.906 (−1.9%) |

**Augmentation is responsible.** Same architecture, same speed (15.9 ms vs 15.7 ms on a T4), same clean
accuracy, and no collapse. Model size only sets the accuracy level (+0.023 mAP50) and gives a slightly
better σ=30; it has nothing to do with noise robustness.

**The first conclusion was premature.** From the Kaggle tables it seemed obvious: clean accuracy is
0.900 vs 0.900 and speed is the same, so robustness comes for free and `s` with noise should become
the default. We downloaded the weights, checked on the demo scene, and the conclusion did not hold.

| Model | MAR20 confidences on the airfield | At conf ≥ 0.4 |
|---|---|---|
| `s` without noise | 0.83, 0.83, 0.67, 0.55 | **4 objects** |
| `s` with noise | 0.78, 0.67, 0.48, 0.34, 0.33 | **3 objects** |

The result reproduced exactly in two independent runs. Noise augmentation makes the model more
conservative on clean data, and some objects slip under the map threshold. mAP50 cannot see this: it
averages over all thresholds, while the operator looks at one.

**Decision: `s` without noise stays the default.** A clean image is the normal case, and there the
baseline shows the operator more. `s` with noise becomes the mode for noisy imagery (same speed,
no loss at σ=15), `m` with noise the archive mode (+0.023 mAP50 for +40% time). All three are listed
in the README with their cost.

**Methodological lesson:** a test-set metric does not replace checking a real scene at the threshold
the operator actually sees. Here one metric (mAP50) showed parity while another (objects above the
threshold) showed a 25% difference.

**Timing note:** the first run in a process is always slower (10.3 -> 7.8 -> 8.8 s on the same scene).
There is no speed difference between `s` and `s` + noise, and there cannot be: the architecture is the same.

## Sep 18: External review: four aircraft at a civilian airport

An outside reviewer pointed at something we had listed as a limitation but not followed through.
Scene 7 is **King Shaka International**, Durban's civilian international airport. MAR20 reported an E-8,
two C-17s and a C-5 there; during review all four were **confirmed** and counted in the overall 0.940,
and the results section presented this as a success. There are no military aircraft there.

**The system contradicted itself with its own measurements.** Sizes in metres had been computed since
Sep 15:

| Detection | Measured | Reference | Deviation |
|---|---|---|---|
| C-5 | 38.8 × 35.6 m | 75.5 × 67.9 m | 48.6% |
| C-17 (id 794) | 28.2 × 26.4 m | 53.0 × 51.8 m | 49.0% |
| C-17 (id 795) | 41.5 × 39.0 m | 53.0 × 51.8 m | 24.7% |
| E-8 | 47.3 × 41.4 m | 46.6 × 44.4 m | 6.8% |

**What we built: `detector/physical.py`.** A table of reference dimensions for the 20 MAR20 types and a
check of the long and short axes with a 25% tolerance. For variable-sweep aircraft (Tu-160, B-1B,
Tu-22M, Su-24) both wing positions are allowed. The verdict (`ok` / `mismatch` / `unknown`) goes into the
GeoJSON, the catalogue, the report and a separate map filter.

**The tolerance was not tuned to the demo.** At 25% and 30% two of the four are flagged, at 20% three.
It was tempting to pick 20% for the nicer "three out of four", but that would be fitting a tolerance to
four objects. We kept 25% and recorded that the borderline C-17 (24.7%) passes. (Refined the same day
into three grades, see below.)

**The main limitation of the method, stated by us:** geometry is a filter, not proof. The E-8 is built
on the Boeing 707 airframe; its 46.6 × 44.4 m is indistinguishable from a civilian wide-body, so it
passes while still being a civilian aircraft. Coverage is also narrow: references exist only for the
MAR20 types, i.e. for **4 of 201 detections**; there is no table for the DOTA classes.

**Relabelling, and two numbers instead of one.** All four got `review_class = plane`, "aircraft, type
not determined": the object is real, the military type is not. We now report separately:

| What is checked | All scenes | Airfield |
|---|---|---|
| the object is really there | 189/201 = 0.940 | 5/7 = 0.714 |
| class or type is correct | 185/201 = **0.920** | 1/7 = **0.143** |

The single averaged 0.940 hid the failure on types. Splitting it is more honest and defensible.

**"Expert review" renamed.** The review was done by one person: 201 objects in 13 min 50 s, about
4 seconds per object. That is visual verification by an operator, not expert assessment, and it is
now described that way everywhere.

**Also done in the same pass:**
- the non-commercial licences (MAR20, Maxar Open Data) are named as a precondition for a pilot in the
  README and the report rather than buried in the licence list;
- small vehicles: all 9 small-vehicle detections have conf 0.25–0.35 and are invisible at the default
  map threshold of 0.4; recorded as a limitation;
- the demo composition is stated plainly: 114 of the 201 objects are storage tanks; it is an oil
  terminal, not a showcase of all classes.

**Lesson.** The limitation had been written down since Sep 16 and still stayed just text, because the
0.940 number did not reflect it. A stated limitation that is not reflected in the metric does not work:
people look at the number.

## Sep 18: Kazakhstan: Sentinel-2 over Atyrau and a measured resolution limit

There is no open imagery of Kazakhstan at 1 m or better (checked on Sep 16), but that is no reason to
have no Kazakhstan data at all. We ran what is open and available.

**Image.** Sentinel-2 L2A `S2B_39TWN_20260914_1_L2A`, 2026-09-14, **0.0%** cloud, EPSG:32639, 10 m/px.
A 2048 px window = 20.5 km around Atyrau (51.8833, 47.1167): the city, the Ural river bend, the port,
oil infrastructure. Sentinel stores every band as a separate COG, so `fetch_demo_scene.py` gained RGB
assembly from several single-band sources (`fetch_bands`) and the `atyrau_s2` preset.

**Result: 3 detections in 4.9 s on 9 tiles:**

| Class | Confidence | Size | Assessment |
|---|---|---|---|
| ground track field | 0.46 | 264×144 m | plausible, a stadium |
| bridge | 0.30 | 152×51 m | plausible, a bridge over the Ural |
| harbor | 0.29 | 701×239 m | **false**, the edge of an evaporation pond |

Not a single storage tank or ship, in an oil city with a river port. The pipeline raised its own
warning: "resolution 10.003 m/px is coarser than 2.0 m/px".

**One run proves little.** Three detections in Atyrau could also mean there are simply no objects of
these classes in the window. So we ran a control experiment on a scene with known content: our Durban
oil terminal, progressively degraded, one model, one threshold of 0.25:

| GSD | Size, px | Total | Storage tanks | Other |
|---|---|---|---|---|
| 0.305 m | 4096×4096 | 154 | **114** | vehicles 15, pools 10, small vehicles 9, harbors 3, ships 2, stadium 1 |
| 1 m | 1250×1250 | 112 | 104 | vehicles 6, ships 2 |
| 2 m | 625×625 | 115 | **113** | ships 2 |
| 5 m | 250×250 | 68 | 66 | ships 2 |
| 10 m | 125×125 | 10 | 5 | "ships" 5 |

**Large objects survive longer than small ones, more strongly than we expected.** Storage tanks hold up
to 2 m almost completely (113 of 114) and 58% remain at 5 m. Small objects vanish at once: pools and
small vehicles already at 1 m, trucks by 2 m. At 10 m only a handful remain, mostly false: five "ships"
where there are two. (The series is not monotonic, 104 at 1 m vs 113 at 2 m; it is a single run.)

**What this changes for a customer.** Instead of "there is no data for Kazakhstan" we have a measured
requirement for the source: **Sentinel-2 (10 m) is unsuitable for the task; 1–2 m imagery is enough to
monitor infrastructure**, which is the class of KazEOSat-1. Ground vehicles need sub-metre imagery. This
answer rests on numbers, not on reasoning; it is an extrapolation from degraded Durban imagery, and we
never ran KazEOSat-1 data.

**Licensing side note.** Copernicus Sentinel data is free, including for commercial use, unlike MAR20
and Maxar Open Data (CC BY-NC). So the only part of the demo free of the NC restriction is the
Kazakhstan one.

## Sep 18: A third verdict: "borderline"

The review exposed a weak spot in the check itself. The second C-17 (41.5 × 39.0 m against the reference
53.0 × 51.8) deviated by 24.7% and passed as valid under the 25% tolerance, with a margin of three tenths
of a percent. The problem is not the pass itself but the question it invites: "why exactly 25%?". The
answer "otherwise it is not caught" is no good, and "it leaves room for shadows, box error and
georeferencing" is right but undermined by exactly this case.

**The check now has three grades instead of a threshold:**

| Deviation | Verdict | On the demo airfield |
|---|---|---|
| up to 15% | size matches | E-8: 6.8% |
| 15–25% | **borderline** | C-17: 24.7% |
| over 25% | type not confirmed | C-17: 49.0%, C-5: 48.6% |

Three of the four aircraft are now flagged, and none is declared wrong without grounds. The verdict goes
into the catalogue, the GeoJSON, a report column and a separate map filter value (⚠ for a mismatch,
≈ for borderline). Both limits are parameters: `--tolerance` and `--borderline` in
`scripts/recheck_sizes.py`.

**What this does not give.** The E-8 matches its reference within 6.8% and passes legitimately: it is
built on the Boeing 707 and is indistinguishable by size from a civilian wide-body. Geometry cannot flag
all four aircraft, and we do not claim otherwise.

**About the catalogue.** All four stay `confirmed`: the object is there, only the type is disputed, and
that is captured by `review_class = plane` in the same record. In the raw `/api/detections` output,
however, `class_name: "C-5"` sits next to `confirmed`; the map with the size-check filter is the clearer
view.

## Sep 29: Aircraft on Sentinel-2 and registry reconciliation: what is possible and what is not

At the online pitch we were asked to find aircraft on Sentinel-2 imagery and reconcile them with an
equipment registry. Before promising anything, we measured.

**Degrading the demo airfield** (two models, threshold 0.25, civilian aircraft 22–47 m long):

| GSD | Aircraft | With a type |
|---|---|---|
| 0.305 m | 6 | 4 |
| 1 m | 5 | 3 |
| 2 m | 5 | 1 |
| 3 m | 5 | 1 |
| 5 m | 4 | 1 (Tu-22, a wrong type) |
| 10 m | **0** | **0** |

- The type holds up to about 1 m. Registry reconciliation needs the type, so it needs 1 m imagery or better.
- The plain fact "there is an aircraft here" lasts longer, up to 3–5 m, at confidence 0.70–0.88.
- At 10 m there is nothing, not even untyped aircraft. For Sentinel-2 this is physics, not a missing
  feature: a 15–22 m fighter covers two pixels.
- The estimate is optimistic: one scene, five civilian aircraft larger than fighters, and artificial
  degradation; a real sensor blurs differently.

**Registry reconciliation: prototype in `detector/inventory.py`.** The registry is a set of airfields
(a WGS84 polygon each) and how many aircraft of each type are listed there. It arrives in the request
`POST /api/scenes/{id}/inventory` or from a file (`scripts/check_inventory.py`) and is never stored.
A registry is customer data and the repository is public, so it only contains a mock example: the
civilian Durban airfield with invented aircraft.

Statuses: matches · not seen in the image · more than listed · type not listed at this airfield ·
type not recognised by the model · airfield outside the image.

**Why reconciliation needs the size check, as seen on the demo:**

| | Without the size check | With the size check |
|---|---|---|
| C-17, 2 listed | 2 detected: "matches" | 1 detected: "not seen: 1" |
| C-5, none listed | "type not listed": **false alarm** | moved to "contradicted by size" |

Without the check the system both reassures wrongly (one of the C-17s cannot physically be one) and
raises a false alarm (an unlisted C-5 that does not exist). With the check both wrong types go to a
separate list, and the report warns that aircraft without a confirmed type may include the ones not seen.

**A bug found by running it.** The first version did not check whether the airfield was covered by the
image. On the Atyrau image the mock airfield (it is in Durban) is not in the frame at all, yet the
reconciliation reported "not seen" for every type, and an operator would conclude that every aircraft
had left the base. Now an airfield outside the image is marked "cannot reconcile", and a partially
covered one comes with a warning about incomplete counts.

**The main limitation for real use.** MAR20 knows 20 types, and many common ones are missing, for
example the Su-27 family. Reconciliation honestly marks such types "not recognised by the model" rather
than "not seen". For a real registry the classifier has to be fine-tuned on the customer's types.

**What to offer a customer:** Sentinel-2 as a free wide-area view of where something changed at
airfields; 1 m imagery for detection and type; registry reconciliation inside the customer's network.

## Sep 29: Recall: the tool exists, and the map threshold turned out to be expensive

**The tool.** The operator marks objects the model missed on the map (`POST /api/scenes/{id}/missed`)
and confirms that the whole scene was inspected. Recall is the share found among everything in the
image: exist = confirmed + misses. A miss is an object absent from the catalogue altogether, so one set
of marks gives recall at both 0.25 and 0.4. The system refuses a miss on top of an existing detection:
that object must be confirmed or reclassified instead. Until the scene is marked as fully inspected, the
numbers are shown as an upper bound.

**Misses must be marked by a person**, and ideally not by the one who built the system; otherwise the
estimate stops being independent.

**The 0.4 map threshold is expensive, visible even before marking misses.** Precision by confidence
band, from the review of 201 detections:

| Confidence | Confirmed | Rejected | Precision |
|---|---|---|---|
| 0.25–0.40 | 36 | 9 | 0.80 |
| 0.40 and above | 153 | 3 | 0.98 |

At 0.4 the operator does not see 36 real objects (17 vehicles, 9 small vehicles, 6 storage tanks,
2 harbors, a swimming pool, a ship) in exchange for hiding 9 false ones. On the terminal, 20 of 34
confirmed objects are visible at 0.4 (59%); on the oil terminal 85%. When a person reviews every
detection anyway this is a bad trade: a false object is rejected in a second, a hidden one is never
seen. Recommendation: default map threshold 0.25–0.3. Not changed before Demo Day because the numbers
in the presentation materials depend on 0.4.

**Why the gain from fine-tuning was not measured.** We planned to fine-tune on the 331 reviewed labels
and dropped it: all of them are the model's own detections confirmed by the operator. If a fine-tuned
model finds something the old one missed, that object is not in the reference labels, and the
evaluation would count it as a false positive, so an improvement would look like a regression. There is
also a skew: training would happen on the airfield and the terminal (72 labels) and testing on the oil
terminal (259, almost all storage tanks). The fine-tuning loop can only be evaluated honestly on a scene
with independent labels, which is what marking misses provides. Learning from misses would also need
boxes, not points.

**A civilian aircraft class: a suitable dataset exists.** RarePlanes (CosmiQ Works, AI.Reverie):
253 WorldView-3 scenes, 112 airports, 14,700 hand-labelled aircraft with a role attribute. Roles: small
civil 8002, medium civil 5132, large civil 1098, all military together 475, i.e. 97% civilian, exactly
what MAR20 lacks. It is open on AWS without registration under CC BY-SA 4.0 (commercial use allowed,
unlike MAR20 and Maxar Open Data). The labels are a "diamond" through the nose, wingtips and tail; an
oriented box should be built along the fuselage axis rather than as the standard minimum-area rectangle,
which would align with a side of the diamond. The tiled imagery is about 3 GB. Not used: adding a
civilian class was out of scope before Demo Day.
