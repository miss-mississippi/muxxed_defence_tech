# Robustness to source image quality

`mar20_s_800.pt` · MAR20 test, 400 images · imgsz 800 · Tesla T4 · `scripts/robustness.py`

Every row is the same subset with one degradation applied. Labels stay the same; only the image changes.

| Degradation | mAP50 | mAP50-95 | mAP50 change |
|---|---|---|---|
| original | 0.900 | 0.781 | — |
| noise σ=15 | 0.371 | 0.300 | −58.8% |
| noise σ=30 | 0.063 | 0.050 | −93.0% |
| blur 5 px | 0.901 | 0.783 | +0.1% |
| blur 9 px | 0.893 | 0.776 | −0.8% |
| JPEG q=30 | 0.866 | 0.748 | −3.8% |
| resolution ÷2 | 0.901 | 0.783 | +0.1% |
| resolution ÷4 | 0.857 | 0.741 | −4.8% |
| brightness ×0.5 | 0.889 | 0.770 | −1.2% |
| haze 30% | 0.901 | 0.780 | +0.1% |

## After the fix: yolo11m trained with noise augmentation

`weights/mar20_m_800_noise.pt` (in the Kaggle output the same file is named `mar20_s_800_noise2.pt`:
the run name is left over from `s`, but the architecture is **m**, trained with `GaussNoise(p=0.5)`),
same protocol: MAR20 test, 400 images, imgsz 800, T4.

| Degradation | mAP50 | mAP50-95 | mAP50 change |
|---|---|---|---|
| original | 0.923 | 0.802 | — |
| **noise σ=15** | **0.918** | 0.800 | **−0.5%** |
| **noise σ=30** | **0.906** | 0.786 | **−1.9%** |
| blur 5 px | 0.919 | 0.801 | −0.4% |
| blur 9 px | 0.901 | 0.787 | −2.4% |
| JPEG q=30 | 0.917 | 0.797 | −0.7% |
| resolution ÷2 | 0.919 | 0.800 | −0.4% |
| resolution ÷4 | 0.883 | 0.770 | −4.4% |
| brightness ×0.5 | 0.927 | 0.807 | +0.4% |
| haze 30% | 0.911 | 0.792 | −1.3% |

### Before and after

| Degradation | before (s, no noise) | after (m, with noise) |
|---|---|---|
| noise σ=15 | 0.371 (−58.8%) | **0.918 (−0.5%)** |
| noise σ=30 | 0.063 (−93.0%) | **0.906 (−1.9%)** |

**The collapse under noise is gone.** A model that lost 93% of its accuracy on a heavily noisy image now
loses 2%. The other degradations are handled more evenly too; the worst is a fourfold drop in
resolution, −4.4%.

## What made the difference: augmentation, not architecture

Two factors changed between the first two tables: architecture (`s` -> `m`) and augmentation
(none -> noise). To separate them, the same protocol was run on `s` trained with noise
(`mar20_s_800_noise.pt`, measured with `notebooks/kaggle_robustness.ipynb`).

| Degradation | mAP50 | mAP50-95 | P | R | mAP50 change |
|---|---|---|---|---|---|
| original | 0.900 | 0.783 | 0.831 | 0.849 | — |
| **noise σ=15** | **0.900** | 0.778 | 0.861 | 0.823 | **−0.1%** |
| **noise σ=30** | **0.890** | 0.769 | 0.838 | 0.827 | **−1.2%** |
| blur 5 px | 0.899 | 0.782 | 0.849 | 0.833 | −0.2% |
| blur 9 px | 0.881 | 0.768 | 0.812 | 0.820 | −2.1% |
| JPEG q=30 | 0.896 | 0.778 | 0.827 | 0.839 | −0.5% |
| resolution ÷2 | 0.898 | 0.780 | 0.861 | 0.816 | −0.3% |
| resolution ÷4 | 0.864 | 0.750 | 0.774 | 0.800 | −4.1% |
| brightness ×0.5 | 0.894 | 0.774 | 0.864 | 0.796 | −0.7% |
| haze 30% | 0.894 | 0.777 | 0.847 | 0.818 | −0.7% |

### Three models side by side

| Degradation | s, no noise | **s, with noise** | m, with noise |
|---|---|---|---|
| original | 0.900 | 0.900 | 0.923 |
| noise σ=15 | 0.371 (−58.8%) | **0.900 (−0.1%)** | 0.918 (−0.5%) |
| noise σ=30 | 0.063 (−93.0%) | **0.890 (−1.2%)** | 0.906 (−1.9%) |

**Answer: it is the augmentation.** Same `s` architecture, same clean-test quality (0.900 vs 0.900),
same speed, and no collapse under noise. The architecture change only sets the accuracy level
(+0.023 mAP50 on the clean test) and holds σ=30 slightly better; it has nothing to do with noise
robustness.

## What follows

**The model is robust** to blur, JPEG compression, halved resolution, low light and thin haze: losses
stay within 5%, and in places there are none. In practice the system survives imagery taken through
haze, recompressed archive images and data from a coarser sensor.

**The baseline model is vulnerable to sensor noise.** At σ=15 mAP50 drops from 0.900 to 0.371, at
σ=30 to 0.063. The cause is training: the augmentations had flips, mosaic and colour shifts but no
noise, so a noisy image looks unfamiliar to the model.

**What was done:** Gaussian noise augmentation was added to training (`scripts/train.py --noise 0.5`,
via the standard ultralytics `augmentations` argument). The measurements above show it is enough:
`s` with noise holds σ=15 without loss while staying the same model in size and speed.

**What this means for the default model.** Robustness is not free, even though mAP50 makes it look so.
On the clean MAR20 test both models score 0.900 and run at the same speed (same architecture, 15.9 ms
vs 15.7 ms on a T4), but on a real clean image the noise-trained model is noticeably more cautious.
Durban demo airfield, the same image, two independent runs with the same result:

| Model | MAR20 confidences | At conf ≥ 0.4 |
|---|---|---|
| `s` without noise | 0.83, 0.83, 0.67, 0.55 | **4 objects** |
| `s` with noise | 0.78, 0.67, 0.48, 0.34, 0.33 | **3 objects** |

mAP50 does not show this: it averages over all thresholds, while the operator sees the map at one
threshold, 0.4. Noise augmentation makes the model more conservative on clean data, and some objects
slip under the threshold.

**So `s` without noise stays the default:** a clean image is the normal case, and there it shows the
operator more. `s` with noise is the mode for noisy imagery: same speed, but no loss at σ=15 instead
of a collapse to 0.371.

`m` with noise (`weights/mar20_m_800_noise.pt`, not in git) remains the archive-processing mode: it buys
+0.023 mAP50 on the clean test and a slightly better σ=30 for +40% scene time (19.0 s vs 13.4 s on the
demo scene with two models).

The 0.900 baseline on the 400-image subset matches 0.899 on the full 2511-image test set, so the subset
is representative.
