# Technical report: automated object identification in satellite imagery

Prepared for Task 3 of the NU STeP × Defense Tech Challenge (Astana, 2026), set by the Military Space
Programmes Centre of the Ministry of Defence of Kazakhstan. This document collects the analysis,
requirements, methodology and results. Raw numbers and the decision log are in
[NOTES.md](../NOTES.md); training plots are in [docs/training](training).

## 1. Purpose and components

The system takes a satellite image, finds objects in it, determines their class and type, georeferences
them and returns the result as a catalogue and as GeoJSON for an external platform.

| Requirement | Where it is implemented |
|---|---|
| Data ingestion and preparation | `POST /api/scenes`, `detector/scene.py` |
| Pre-processing | `detector/scene.py`: tiling, bit-depth stretch, band selection, empty-tile filtering |
| Automated detection | `detector/model.py`, `detector/registry.py` |
| Classification | second model in the registry (equipment types), refine mode |
| Spatial delineation | oriented bounding boxes, `detector/schema.py` |
| Visualisation | `web/index.html` (Leaflet) |
| Reporting | `GET /api/scenes/{id}/report` |
| Integration interfaces | REST + GeoJSON, `POST /api/detect` |
| Operator review | `PATCH /api/detections/{id}`, the map, `scripts/export_reviewed.py` |

## 2. Methods

**Why oriented boxes (OBB) rather than axis-aligned ones.** In overhead imagery objects lie at arbitrary
angles and close together: aircraft on an apron, ships at a berth, vehicles in a parking lot. An
axis-aligned box around such an object is half background, neighbouring boxes overlap, and duplicate
suppression removes real objects. An oriented box also gives the object's heading, which is a quantity
of its own for situation analysis; the system reports it as the azimuth of the long axis.

**Choice of detector family.** Two-stage detectors (Oriented R-CNN, ReDet) are slightly more accurate on
DOTA but slower and harder to deploy. One-stage detectors (YOLO-OBB, RTMDet-R) give up a fraction of a
percent of mAP, process a tile in tens of milliseconds and deploy as a single weights file. For a system
that has to scan thousands of square kilometres throughput decides, so we chose YOLO11-OBB.

**Why transfer learning.** Training a detector from scratch needs tens of thousands of labelled images
and weeks of compute. Weights pretrained on DOTA already encode overhead-imagery objects; fine-tuning for
new classes takes tens of minutes on two GPUs and needs thousands, not tens of thousands, of examples.
This is a deliberate engineering choice, not a shortcut.

**Two-level recognition.** A general model (DOTA, 15 classes) handles detection: ships, storage tanks,
vehicles, harbours, bridges, aircraft. A specialised model (MAR20, 20 military aircraft types) handles
type recognition and runs only on top of the general model's detections. This split lets us add types
without retraining the general model and without losing the other classes when a new specialisation is
added.

**Checking classification with physics.** The system is georeferenced, so every object has not only a
class but also a size in metres. That enables a check a non-georeferenced detector cannot do: the
measured dimensions are compared with the reference dimensions of the claimed type
(`detector/physical.py`). There are three verdicts rather than one threshold: up to 15% deviation is
explained by box error, shadows and georeferencing accuracy ("size matches"); 15–25% is "borderline";
beyond that the type is "not confirmed by size". The verdict is shown in the catalogue, in the report and
as a separate map filter.

A binary threshold misleads right at the boundary: on the demo scene one C-17 deviated from its reference
by 24.7%, and with a hard 25% threshold it would have passed as valid while being a quarter off the
reference size. The third grade handles this case and answers the question "why exactly 25%".

The check is independent of the neural network, so it catches errors where the model is confident: a
C-17 at confidence 0.83 measured 28.2 m against the reference 53.0 m, a 49.0% deviation and the verdict
"type not confirmed". The largest size error (a C-5 at 38.8 m against 75.5 m) had a low confidence of
0.55. So confidence and size plausibility are independent, and the check adds a second signal where the
first one says nothing. Its limits are stated upfront: matching dimensions do not prove the type. The
E-8 is built on the Boeing 707 airframe, matches its reference within 6.8% and passes while being a
civilian aircraft. The reference table currently covers 20 aircraft types.

## 3. Satellite imagery specifics and input requirements

| Factor | Effect | How it is handled |
|---|---|---|
| Spatial resolution | the limit is measured, not estimated: storage tanks hold up to 2 m, vehicles disappear at 1–2 m, at 10 m only a handful of detections remain | requirement of ≤1 m, a warning in the result when GSD > 2 m, the measurement table in section 7 |
| Bit depth | remote sensing data is usually 11–16 bit | one 2–98 percentile stretch for the whole scene |
| Spectral bands | the models are trained on RGB | R, G, B are selected from the GeoTIFF colour interpretation; panchromatic is replicated |
| Illumination and haze | reduce contrast | robustness measured with `scripts/robustness.py` |
| Viewing angle | off-nadir views distort shape | the demo scenes were captured ~28° off-nadir |
| Scene size | scenes are tens of thousands of pixels across | 1024 px tiles with 200 px overlap and merging at the seams |
| Missing data | black borders, nodata | tiles are dropped using the mask and raw values |

## 4. System requirements

- **Data:** georeferenced GeoTIFF (any CRS), RGB or panchromatic, GSD ≤1 m, 8–16 bit. Non-georeferenced
  images are also accepted; the output is then in pixels.
- **Accuracy:** target mAP50 of at least 0.85 on the target class set; achieved 0.899 on the official
  MAR20 test set.
- **Throughput:** at least 1000 km²/h on one machine without a server GPU. Achieved about 3600 km²/h at
  0.5 m/px on an Apple M1 laptop, and about 1050 km²/h on its CPU. Measured on a synthetic scene with one
  model; at a realistic 0.3 m/px this is ~1300 km²/h.
- **Scalability:** scene processing is split into independent tiles, models are loaded by a registry,
  and adding a model is one environment variable.
- **Information security:** weights and data stay inside the network; the system makes no external calls
  while processing. The only external request is basemap tiles, so the interface has a "no basemap" mode
  for isolated networks.
- **Reproducibility:** library versions are pinned, training uses a fixed seed, and 60 automated tests
  cover tiling, merging, georeferencing, the data converter, the API, the size check, registry
  reconciliation and recall.

## 5. Training data methodology

1. Source labels are converted to YOLO-OBB: four corners normalised to the image size
   (`scripts/prepare_mar20.py`).
2. Input quality control: every image must have a label file, the image size is read from the file when
   the label has a zero size, and coordinates are clipped to the frame. Per-class and per-split
   statistics are saved next to the dataset.
3. Splits: train and validation are carved out of the source train set; the test set stays untouched.
   Metrics are computed on it and epoch selection never sees it.
4. Augmentations suit overhead imagery: flips on both axes, mosaic, optional rotation, mosaic disabled for
   the last epochs; Gaussian noise optionally (see section 7).
5. Updating: detections confirmed by the operator become new training material
   (`scripts/export_reviewed.py`). Only fully reviewed tiles enter the dataset; otherwise an unreviewed
   object would become "background" and teach the model to miss it.

## 6. Evaluation methodology

- **Metrics:** mAP50 and mAP50-95, per-class precision and recall, image and scene processing time,
  throughput in km²/h, robustness to image degradation.
- **Data:** the dataset's official test split; the demo scenes are evaluated by operator review in the
  interface because they have no reference labels. For aircraft types this is complemented by the
  network-independent size check (`detector/physical.py`).
- **Tools:** `scripts/eval.py` (metrics), `scripts/benchmark.py` (throughput), `scripts/robustness.py`
  (robustness).
- **Reproduction:** commands are in the [README](../README.md); results are saved to `outputs/` as JSON and
  Markdown.

## 7. Results

- Military aircraft type recognition: mAP50 **0.899**, mAP50-95 **0.786**, precision 0.879, recall 0.841
  on the 2511-image MAR20 test set. Per-class breakdown in [NOTES.md](../NOTES.md).
- **Architecture comparison.** Two models were trained under identical conditions: yolo11s (mAP50 0.899,
  15.7 ms/image) and yolo11m (mAP50 0.904, 43.7 ms/image). A 0.5% accuracy gain for a 2.8× slowdown, so
  yolo11s is the default and yolo11m stays an option for archive processing where throughput is not
  critical. Per class the models differ more than on average (F-22: 0.589 vs 0.717 in favour of m;
  F-16: 0.747 vs 0.649 in favour of s), which points to a lack of examples of these types rather than a
  lack of model capacity.
- Throughput: a **synthetic** 8192×8192 scene at 0.5 m/px (16.8 km²) takes 16.8 s on an Apple M1 (MPS)
  and 57.4 s on its CPU, with one DOTA model and no MAR20. Throughput scales with the square of the
  resolution: ~3600 km²/h at 0.5 m/px and ~1300 km²/h at a realistic 0.3 m/px.
- Real imagery (Maxar Open Data, Durban, 0.305 m/px): aircraft with types on the airfield, ships and
  vehicles on the container terminal, 105 storage tanks on the oil terminal (at conf ≥ 0.4).
- **Operator review:** all 201 detections on three scenes were reviewed. Two different quantities are
  reported, because a single one mixes them and hides a failure:

  | What is checked | All scenes | King Shaka airfield |
  |---|---|---|
  | the object is really there | 189/201 = **0.940** | 5/7 = 0.714 |
  | class or type is correct | 185/201 = **0.920** | 1/7 = **0.143** |

  The airfield gap has a direct explanation: King Shaka is a **civilian** international airport, MAR20
  has no civilian classes, so all four aircraft got military types (E-8, C-17 ×2, C-5). The objects are
  found correctly, the types are not reliable; during review they were relabelled `plane`, "aircraft,
  type not determined". The weakest class by object presence is ships (0.500): fragments of a ship under
  port cranes.
- **How the review was done.** One operator, 201 objects in 13 minutes 50 seconds, about 4 seconds per
  object. This is on-screen verification, not expert assessment: the team has no certified image
  analysts. The number measures precision; **recall is unknown**, because objects the model missed never
  enter the review by construction.
- **The fine-tuning loop works on real data:** reviewed detections were exported into a dataset of
  51 tiles, with 331 labels for the general model and 5 for the type model.
- Robustness to image quality: blur, JPEG compression, halved resolution, low light and haze cost no more
  than 5% mAP50. This is a **relative** loss, not percentage points; the measurement used a 400-image
  subset of the test set (its baseline is 0.900 vs 0.899 on all 2511). Sensor noise drops the metric from
  0.900 to 0.371 (σ=15) and 0.063 (σ=30). Table and conclusions:
  [docs/training/robustness.md](training/robustness.md).
- **The resolution limit is measured, not estimated.** The oil-terminal demo scene (0.305 m/px) was
  progressively degraded and processed with one model at one threshold of 0.25:

  | GSD | Size, px | Total | Storage tanks | Other |
  |---|---|---|---|---|
  | 0.305 m | 4096×4096 | 154 | **114** | vehicles 15, pools 10, small vehicles 9, harbours 3, ships 2, stadium 1 |
  | 1 m | 1250×1250 | 112 | 104 | vehicles 6, ships 2 |
  | 2 m | 625×625 | 115 | **113** | ships 2 |
  | 5 m | 250×250 | 68 | 66 | ships 2 |
  | 10 m | 125×125 | 10 | 5 | "ships" 5 |

  Large objects survive longer than small ones: storage tanks are kept almost completely up to 2 m
  (113 of 114), and 58% remain at 5 m. Small and medium vehicles disappear first: pools and small
  vehicles already at 1 m, trucks by 2 m. At 10 m only a handful remain, mostly false: five "ships"
  where there are two.
- **A run on Kazakhstan territory.** A Sentinel-2 L2A image over Atyrau (S2B, 2026-09-14, 0.0% cloud,
  10 m/px, a 20.5 km window: city, the Ural river, port, oil infrastructure) went through the same
  pipeline: 9 tiles, 4.9 s, **3 detections**: a stadium (0.46), a bridge over the Ural (0.30) and a
  701×239 m "harbour" at 0.29 on the edge of an evaporation pond, i.e. a false one. No storage tanks, no
  ships. The system warned on its own: "resolution 10.003 m/px is coarser than 2.0 m/px". The conclusion
  for a customer: **Sentinel-2 is unsuitable for this task, while 1–2 m imagery is enough to monitor
  infrastructure**; per the table above, storage tanks survive at 2 m almost completely. Caveat: this is
  derived from degraded Durban imagery; we never ran KazEOSat-1 data, so it is a statement about the
  required resolution class, not a tested satellite.
- **Aircraft and resolution.** Degrading the demo airfield with two models at a threshold of 0.25: the
  type is determined for 4 aircraft at 0.3 m, 3 at 1 m and one at 2 m; the plain fact "there is an aircraft
  here" holds up to 3–5 m; at 10 m nothing is found. Registry reconciliation by type needs ~1 m imagery,
  and on Sentinel-2 aircraft are not observable at all: a fighter there covers two pixels. The estimate
  is optimistic: five civilian aircraft larger than fighters, and artificial degradation.
- **Working on the weakness found.** Gaussian noise augmentation was added to training
  (`train.py --noise`), and two more models were trained. All four runs on the MAR20 test set:

  | Architecture | Noise | mAP50 | mAP50-95 | Inference |
  |---|---|---|---|---|
  | yolo11s | no | 0.899 | 0.786 | **15.7 ms** |
  | yolo11s | yes | 0.898 | 0.784 | 18.3 ms |
  | yolo11m | no | 0.904 | 0.795 | 43.7 ms |
  | yolo11m | yes | **0.915** | **0.804** | 40.4 ms |

  Noise augmentation improved the large model (+0.011 mAP50) and did not change the mean metric of the
  small one. The best accuracy belongs to yolo11m with noise, but it is 2.6× slower; for scanning
  thousands of km² throughput matters more than 1.6% mAP50.

  **The collapse under noise is fixed, and we know what fixed it.** The same robustness protocol was run
  on three models, separating the two factors that had changed together, model size and augmentation:

  | Degradation | s, no noise | s, with noise | m, with noise |
  |---|---|---|---|
  | original | 0.900 | 0.900 | 0.923 |
  | noise σ=15 | 0.371 (−58.8%) | **0.900 (−0.1%)** | 0.918 (−0.5%) |
  | noise σ=30 | 0.063 (−93.0%) | **0.890 (−1.2%)** | 0.906 (−1.9%) |

  Augmentation is responsible: same architecture, same speed, and no collapse. Model size only sets the
  accuracy level (+0.023 mAP50) and has nothing to do with noise robustness.

  **A check on a real scene refined the decision.** On a clean image the noise-trained model is noticeably
  more cautious: on the demo airfield at the working map threshold conf ≥ 0.4 it shows three objects
  against four for the baseline (confidences 0.78/0.67/0.48/0.34/0.33 vs 0.83/0.83/0.67/0.55, reproduced
  in two independent runs). The mean metric cannot see this: mAP50 averages over all thresholds, while
  the operator works at one. So the default remains yolo11s without noise; the noise-trained model is
  switched on with one environment variable for noisy imagery, and yolo11m with noise is the
  archive-processing mode.

  The full "measure -> hypothesis -> change training -> re-measure -> check on real data" loop was
  completed and is backed by numbers ([docs/training/robustness.md](training/robustness.md)).
- Training curves, the PR curve and the confusion matrix are in [docs/training](training).

## 8. Limitations

1. **The specialised model is trained only on military types; MAR20 has no civilian classes.** At the
   King Shaka demo airfield, Durban's civilian international airport, all four aircraft got military
   types (E-8, C-17 ×2, C-5), so type accuracy there is 1/7 = 0.143 against an object-presence accuracy
   of 5/7 = 0.714. The size check flags three of the four: two as not confirmed (49.0% and 48.6%) and one
   as borderline (24.7%). The fourth, the E-8, matches its reference within 6.8% and passes: by size it is
   indistinguishable from a civilian wide-body. The system is meant for military airfields; on civilian
   ones its type output is unreliable, and the operator sees this from the model tag and the size
   verdict.
2. Open imagery of Kazakhstan at 1 m or better does not exist: Maxar Open Data has no events in the
   country, and OpenAerialMap has no captures of key airfields or ports. The demo uses open Durban
   imagery; the system does not depend on the source and accepts any GeoTIFF.
3. When merging duplicates, the box not truncated by a tile edge wins over the more confident one.
4. Evaluation on the demo scenes is visual: there are no reference labels for them. One operator did the
   review, 201 objects in 13 minutes 50 seconds, about 4 seconds per object. This is on-screen
   verification, not expert assessment. The number measures precision; recall on these scenes is still
   unknown. The tool to measure it exists: the operator marks objects missed by the model on the map,
   and recall is computed per class at thresholds 0.25 and 0.4. Misses should be marked by a person,
   otherwise the estimate is not independent.
5. The size check covers only the 20 MAR20 aircraft types: of the 201 demo detections, a reference
   applies to 4. For the DOTA classes (ships, storage tanks, vehicles) there is no reference table, and the
   verdict stays "not checked".
6. **The demo data is non-commercial.** MAR20 and Maxar Open Data are licensed CC BY-NC. A pilot
   deployment needs licensed imagery and a commercially usable fine-tuning dataset; this is a
   precondition, not a formality.
7. Small objects come out with low confidence: all 9 small-vehicle detections on the demo scenes are at
   0.25–0.35 and are not shown at the default map threshold of 0.4. To review ground vehicles, the
   threshold is lowered to 0.25 at the cost of more false positives.

## 9. Recommendations for development and integration

- **More object types:** a new specialised model is added to the registry without retraining the
  general one. First candidates: ship types and ground vehicles.
- **Model improvement:** the closed loop "detection -> operator review -> dataset export -> fine-tuning"
  already works; accumulating reviewed regional scenes improves accuracy without new data sources.
- **Extending the physical check:** the reference table currently covers 20 aircraft types. The same idea
  applies to ships (hull length), storage tanks (diameter) and ground vehicles, giving a
  network-independent check wherever the image scale is known.
- **Default map threshold.** According to the review, detections at confidence 0.25–0.4 are 80% precise
  (36 confirmed, 9 rejected), and the 0.4 threshold hides those 36 real objects from the operator. When a
  person reviews every detection, lowering the threshold to 0.25–0.3 is reasonable.
- **Registry reconciliation** (`detector/inventory.py`, `POST /api/scenes/{id}/inventory`): the prototype
  compares detected aircraft with what is listed at an airfield, taking into account the size check,
  airfield coverage by the image and resolution. The registry is passed in the request and never stored,
  so it stays inside the customer's network. For real use the classifier is fine-tuned on the customer's
  types: many common types are absent from MAR20, and reconciliation honestly reports them as "not
  recognised".
- **Integration with a remote sensing platform:** the synchronous `POST /api/detect` accepts an image and
  returns GeoJSON; the detection catalogue is available via `GET /api/detections` with filters by class,
  model, confidence, review status and area.
- **Isolated networks:** disable external basemaps, host the system on a secured server, keep the weights
  local.
- **Scaling:** tiles are processed independently, so the system scales with the number of GPUs; the scene
  queue can be moved into a separate service without changing the rest of the code.
