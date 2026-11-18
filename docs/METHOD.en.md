# Method: deciding "same person" with deep face recognition

This document explains the algorithm pipeline, the parameters used by
`face-match-lab`, and the reasoning behind each choice.

> **About the numbers**: values quoted as "measured" come from the **NASA
> public-domain sample photos** shipped with this repository and can be
> reproduced in one command: `python src/face_compare.py --samples`.
> For sources and licensing see [`samples/README.en.md`](../samples/README.en.md).
>
> 中文版: [METHOD.md](METHOD.md)

---

## 1. Pipeline overview

```
Input: group photo A (multiple people) + reference photo B (single) [+ third angle C]
   │
   ├─(1) Multi-scale face detection  SCRFD / YuNet  ──► list of face boxes
   │                                            │
   ├─(2) Reference subject        largest face     │
   │                                            ▼
   ├─(3) Group target             retrieval / geometric strategy
   │                                            │
   ├─(4) Feature extraction        ArcFace / SFace  ──► 512-d / 128-d L2-normalised embeddings
   │                                            │
   ├─(5) Similarity               Euclidean distance + cosine similarity
   │                                            │
   └─(6) Threshold verdict + confidence mapping ──► same person / different person
```

---

## 2. Face detection

### 2.1 Primary detector: SCRFD (insightface)

- Model bundle: `buffalo_l`, detection model `det_10g.onnx`
- Input size: `det_size=(640, 640)`; aspect ratio preserved via `letterbox`
- Confidence threshold: `det_model.det_thresh = 0.45`
  - The 0.5 default under-recalls small, distant, backlit, or profile faces;
    relaxing to 0.45 recovers roughly 3–5% more targets

### 2.2 Multi-scale enhancement: `det_sizes()`

In large group photos, distant faces occupy very few pixels and a single scale
misses them. Two scales are run independently:

| Scale | Target long edge | Typical use |
|---|---|---|
| Scale A | 640 | close-ups, medium shots |
| Scale B | 1600 | wide group shots, full scenes |

**Constraint**: SCRFD's anchor stride is 32, so input dimensions must be a
multiple of 32 — otherwise `onnxruntime` throws `Invalid input shape`.
`det_sizes()` applies a `ceil(x/32)*32` correction to every scale.

**Deduplication**: results from both scales are merged with IoU < 0.4 greedy
NMS, so the same face is never counted twice.

### 2.3 Fallback detector: YuNet (OpenCV Zoo)

When insightface is unavailable (e.g. no compiler), the pipeline switches
automatically:

- Detection: `cv2.FaceDetectorYN_create`
- Recognition: `cv2.FaceRecognizerSF_create`
- Output: 128-d embedding

The onnx models are only downloaded when `not ARCFACE_OK`, avoiding pointless
network traffic.

---

## 3. Locating the target face

### 3.1 Two families of strategy

Strategies are classified by whether they consult the reference photo. This
classification determines how the resulting verdict should be interpreted.

| Family | Strategies | Looks at reference? | Question answered |
|---|---|---|---|
| **Retrieval** | `retrieve` | yes | "Is this person in the group?" |
| **Geometric** | `largest` / `front-right` / `white-shirt` / `index:N` | no | "Is the person at this position the one I mean?" |

Geometric strategies never look at the reference photo when choosing a face, so
their verdict is **independent evidence** and can be cross-checked against the
retrieval result — agreement between the two makes the conclusion stronger.

### 3.2 Retrieval: `retrieve` (default)

Among all detected faces in the group photo, the one whose embedding has the
**highest cosine similarity** to the reference subject is selected.

This is by definition the Top-1 of the traversal ranking, so under this
strategy:
- `comparison` reflects whether *any* candidate in the group exceeds the threshold
- `ranking_margin` (Top-1 minus Top-2) reflects the **uniqueness** of the match

### 3.3 Geometric criteria

**`largest`** — the face box with the largest area.

**`front-right`** — three constraints applied jointly:

1. **Vertical position**: face centre `cy / H > 0.52` counts as the lower half
2. **Horizontal position**: within that half, the largest `cx` (rightmost) wins
3. **Fallback**: if the lower half is empty, the globally rightmost face is used

**`white-shirt`** — adds a clothing check on top of the `front-right` candidate
ordering:

- Shoulder region is measured in HSV; saturation `S < 70` and value `V > 140`
  indicates white clothing
- The sampled shoulder strip extends 0.35 × box height below the face box

**`index:N`** — directly selects the N-th face in the group photo (in detection
order, 1-based). Useful for controlled evaluation where the answer is known.

### 3.4 Fallback chain

When a geometric strategy cannot be satisfied, it degrades uniformly to
"largest face" so the pipeline never stalls on an unusual composition.
`select_target()` returns a `matched_by` field recording which criterion
actually fired, so any decision can be traced after the fact.

### 3.5 Reference subject and third-angle person

- **Reference subject**: the largest face in the reference photo (in a close-up
  the subject is nearest the lens)
- **Third-angle person**: the face in the third photo most similar to the
  reference subject

> Note: the third-angle person is located by *retrieval*, unlike the geometric
> strategies above. This is deliberate — the third photo exists to provide
> independent evidence, and its identity must by definition be established from
> the reference. The group photo, by contrast, supports both ways of choosing.

---

## 4. Features and similarity

### 4.1 Embeddings

- ArcFace (`w600k_r50`): **512-d**, L2-normalised on output
- SFace (OpenCV Zoo): **128-d**, L2-normalised on output

For normalised vectors `a`, `b`:

```
cosine_similarity = a · b                       ∈ [-1, 1]
euclidean_distance = ||a - b||                   ∈ [0, 2]
```

After normalisation the two are strictly equivalent:

```
cosine_similarity = 1 - (euclidean_distance² / 2)
```

Computing either therefore yields both, and the two reported values are always
mutually consistent — this is the source of the internal consistency in the
reports.

### 4.2 Threshold selection

| Model | Cosine threshold | Equivalent Euclidean | Basis |
|---|---|---|---|
| ArcFace | **0.40** | **1.095** | equivalent form of `face_recognition`'s common 0.6 threshold (which is defined on L2-normalised features) |
| SFace | **0.363** | **1.169** | SFace's officially recommended threshold |

> Note: `face_recognition`'s 0.6 is defined on **L2-normalised** features,
> corresponding to cosine 0.4 / Euclidean 1.095. This project states everything
> in cosine form so values can be compared across models.

### 4.3 Confidence mapping

A logistic mapping turns cosine similarity into an intuitive percentage:

```
confidence = 100 / (1 + exp(-k * (cos - threshold)))      k = 12.0
```

- `cos = threshold` → 50%
- `cos = threshold + 0.2` → 91.0%
- `cos = threshold - 0.2` → 9.0%

This is a **monotonic illustration** of verdict probability, not a statistical
posterior. Values near the threshold are sensitive; values far from it
saturate quickly.

---

## 5. Triangulated verification

A single pairwise comparison is easily disturbed by detection error, so the
project locates the same individual across **three photos** to form a
triangular evidence chain:

```
        A：group-photo target face
       / \
      /   \
 B：reference   C：third angle
      \   /
       \ /
   all three pairs agree
```

`src/triangulate.py` emits:

- `pairs` — distance / cosine / confidence / verdict for each of the three pairs
- `all_consistent` — whether all three agree
- `ranking_vs_third` — similarity ranking of every face in the group against
  the third-angle person

**Measured** (NASA public sample): A–B 0.5645, A–C 0.7105, B–C 0.5650 — all
three consistent.

---

## 5.1 End-to-end self-test (`scripts/selftest.py`)

Turns "it runs" into an **assertable regression test**. Each check returns
PASS / FAIL / SKIP; any FAIL produces a non-zero exit code, so it plugs
straight into CI.

| ID | Check | Assertion |
|---|---|---|
| T1 | Main verdict | Group retrieval Top-1 is judged "same person" |
| T2 | Uniqueness | Top-1 − Top-2 cosine gap > `MIN_MARGIN` (0.05) |
| T3 | Negative control | The highest similarity among *other* people stays below threshold |
| T4 | Pass count | Exactly **1** face in the group crosses the threshold |
| T5 | Triangulation | A-B / A-C / B-C all agree |
| T6 | Reference robustness | Swapping in the third photo as reference finds **the same face** |

T3 and T4 are two forms of the same negative control: T3 asks whether the
second-best candidate also crosses the line, T4 asks whether the total number
of crossings is exactly one. Together they rule out both "a lucky high score"
and "everything scores high" degeneracies.

T5 / T6 require a third photo; without one they are marked SKIP rather than
failed.

**Measured** (NASA public sample): all 6 checks PASS.
**Measured** (17-person real group photo): all 6 checks PASS — Top-1 cosine
0.5370, margin 0.4610.

The self-test accepts an external data directory, so you can validate against
your own photos without committing them:

```bash
python scripts/selftest.py --samples            # bundled public sample
python scripts/selftest.py --data-dir data      # your own photos
python scripts/selftest.py --group a.jpg --reference b.jpg --third c.jpg
```

---

## 6. Traversal and ranking

The pipeline traverses **every face** in the group photo and emits a ranking
sorted by cosine similarity, descending.

Beyond answering "who matched best", the ranking quantifies the **uniqueness**
of the match through `ranking_margin` (Top-1 minus Top-2):

| Reference | Top-1 | Top-2 | Gap |
|---|---|---|---|
| Glenn portrait | 0.5645 | 0.1831 | **0.3813** |
| Cooper portrait (negative control) | 0.4380 | 0.1395 | **0.2985** |

Both gaps are far larger than the threshold boundary, indicating a decisive
result with no borderline jitter.

> **Why the negative control matters**: swap the reference photo and the Top-1
> result moves to a *different person who genuinely exists* in the group photo.
> This proves the ranking reflects real identity similarity rather than
> "always give some fixed face a high score".
> The assertion is enforced by `scripts/selftest.py` (T3 / T4) and runs in CI.

---

## 7. Report structure and privacy handling

JSON reports are redacted before being written to disk
(`face_compare.anonymize_ranking()`).

### Fields kept

| Field | Contents |
|---|---|
| `backend` / `embedding_dim` | backend name and embedding dimension |
| `threshold` | cosine threshold and its equivalent Euclidean distance |
| `source` | detected face count, image dimensions (**no paths**) |
| `target` | strategy, criterion that fired, position label, shoulder HSV stats (**no coordinates**) |
| `comparison` | Euclidean distance, cosine similarity, confidence, verdict |
| `ranking` | anonymous index + position label + similarity metrics, sorted by similarity |
| `ranking_margin` | cosine gap between Top-1 and Top-2 |

### Fields removed

| Field | Why it is removed |
|---|---|
| `bbox` / `center` | pixel coordinates of face boxes; released with the original photo they reveal where every person stands |
| `path` | original file paths, leaking local directory structure and file names |

Coordinates are used in memory only for visualisation and cropping — they never
reach any on-disk artefact. Correspondingly, `src/plot_results.py` labels chart
axes with "anonymous index + position label" and never pixel coordinates. CI
automatically verifies that generated reports contain none of these fields.

---

## 8. Reproducibility

| Factor | Effect | Control |
|---|---|---|
| Detector version | affects box placement and recall | pinned `insightface==0.7.3` + `buffalo_l` |
| Detection scale | affects small-face recall | fixed at 640 / 1600 |
| Feature model | directly determines the threshold | ArcFace threshold fixed at 0.40 |
| Image rescaling | affects embedding precision | detector letterboxes internally; features are extracted from 112×112 aligned crops |

Given the same input images and the same model version, results are exactly
reproducible (the embedding is a deterministic function). The bundled NASA
sample has been verified to produce identical numbers across repeated runs.

---

## 9. Data preparation and path configuration

No user data path is hard-coded. Resolution order:

```
command-line argument  >  environment variable  >  src/config.py default
```

| Environment variable | Purpose | Default |
|---|---|---|
| `FACE_DATA_DIR` | data root | `<repo>/data` |
| `FACE_SAMPLES_DIR` | public sample directory | `<repo>/samples` |
| `FACE_OUT_DIR` | output root | `<repo>/results` |
| `FACE_CROP_DIR` | crop directory | `<repo>/results/crops` |
| `FACE_REPORT` | main report path | `<repo>/results/report.json` |
| `FACE_TRI_REPORT` | triangulation report path | `<repo>/results/triangulation.json` |
| `FACE_CHART_DIR` | chart directory | `<repo>/assets/charts` |

`scripts/prepare_data.py` validates the `data/` structure and copies the public
samples into place; see the "Using Your Own Photos" section of the README.
