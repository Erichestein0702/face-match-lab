# samples/ — public sample photos

This directory holds **publicly redistributable** sample photographs. They let
anyone run the full pipeline with zero setup, and they provide the
**reproducible** data source behind the measured numbers in the documentation.

> 中文版: [README.md](README.md)

---

## Sample inventory

| File | Content | NASA asset ID | Source |
|:---|:---|:---|:---|
| `mercury7_group.jpg` | Mercury Seven crew portrait (7 faces — the retrieval input) | `s62-08774` | <https://images.nasa.gov/details/s62-08774> |
| `glenn_portrait.jpg` | Official portrait of John H. Glenn (single face — the reference) | `S62-05540` | <https://images.nasa.gov/details/S62-05540> |
| `glenn_spacesuit.jpg` | John H. Glenn in a Mercury spacesuit (third angle — for triangulation) | `S87-41328` | <https://images.nasa.gov/details/S87-41328> |
| `cooper_portrait.jpg` | Official portrait of L. Gordon Cooper (**negative control** — not in the group photo) | `s64-31847` | <https://images.nasa.gov/details/s64-31847> |

---

## Licensing status

All four are **NASA (U.S. National Aeronautics and Space Administration) image
library** assets, produced by federal government employees in the course of
official duties. They are therefore **public domain** works: free to use,
modify, and redistribute without further permission or attribution obligation.

> NASA media usage guidelines: <https://www.nasa.gov/nasa-brand-center/images-and-media/>

Images have been rescaled along the long edge and re-compressed (quality 88) to
keep the repository small. Their content is otherwise unmodified.

---

## Why these four

Together they form a **self-contained control set with known answers**, which
exercises every stage of the algorithm:

| Check | Expected behaviour |
|:---|:---|
| Detection | `mercury7_group.jpg` should reliably yield **7** faces (including 3 small faces in the back row) |
| Retrieval | with `glenn_portrait.jpg` as the reference, Glenn himself should rank **Top-1** |
| Uniqueness | the Top-1 − Top-2 cosine gap should be substantially above 0, proving the match isn't "just picking something" |
| Triangulation | the three faces belonging to Glenn should pairwise agree as "same person" |
| Negative control | swapping in Cooper's portrait should make Top-1 **switch to Cooper** (a different person in the back row) |

The negative control is the most valuable part of this set: it proves the
ranking reflects **genuine identity similarity**, rather than "always give the
highest-scoring face a good score".

```bash
# Negative control: with Cooper as the reference, Top-1 should switch
python src/face_compare.py --samples --reference samples/cooper_portrait.jpg
```

---

## Usage

```bash
# Use the bundled samples directly — no preparation needed
python src/face_compare.py --samples
python src/triangulate.py --samples

# Or copy them into data/ and follow the standard flow
python scripts/prepare_data.py --use-samples
python src/face_compare.py --data-dir data
```

---

## Privacy statement

This directory contains **no private photographs of any user or third party**.
The only real face images in the repository are the NASA public-domain assets
listed above. Your own photos must go in `data/`, which is excluded by
`.gitignore` and can never be committed.
