# People in hats and caps: training photos that are not helmets

**Why.** On real video from other sites (Phase 6, `docs/phase6_results.md`), a bricklayer in a
baseball cap was never alerted: the model took his cap for a helmet. On Open Images photos of
people in hats, the model in use (`ppe4_yolo26n_finetune`) calls about one hat in four a helmet:

| Held-out test photos (version 1: 449 photos, 660 hats, 634 people) | Model in use |
|---|---|
| Hats with a helmet box on them (at the rule's 0.25) | 25.2 % |
| People in hats judged "helmet worn" by the rule (one frame) | 27.8 % |
| For comparison: bare heads judged "helmet worn" (Phase 1 test photos) | 2.4 % |

Its training data (datasets/ppe4) has hard hats and bare heads, but almost no other headwear.

**What is here.** `openimages_caps.json` (version 2, 29 Sep 2026) lists 1,739 photos from Google's
[Open Images V7](https://storage.googleapis.com/openimages/web/index.html), each with its download
address, SHA-256, labels, hats and credit. `scripts/prepare_caps_data.py` (or
`bash scripts/mac_caps.sh data`) downloads them (~570 MB) and builds `datasets/ppe4caps`.

| Split | Photos | Hats | What for |
|---|---|---|---|
| train | 1,043 | 1,616 | added to ppe4's training photos: 693 whose hat the model in use calls a helmet (>= 0.10), 350 it gets right |
| val | 248 | 346 | added to ppe4's validation photos, so training stops when hats are called helmets again |
| test | 448 | 661 | a separate list (`test_caps`), never trained on: how often a model calls a hat a helmet |

Version 1 (28 Sep) had 2,100 photos: 700 "easy" training photos instead of 350, and 13 photos
that a second review left out. Why it changed: `docs/caps_results.md`.

**Licence.** Boxes: CC BY 4.0, Google LLC. Photos: listed by Open Images as CC BY 2.0 by their
Flickr authors; each photo's author and page are in the manifest, and `datasets/ppe4caps/credits.csv`
lists every photo used.

## How they were chosen (28 Sep 2026, in the cloud: `scripts/select_caps.py`)

1. **Open Images' own boxes** (validation, test, and the training split's 2.26 GB box file read
   as a stream): 15,619 photos with a "Hat", "Sun hat", "Fedora", "Cowboy hat" or "Sombrero" box.
   Kept: 10,351 where every hat is worn by a labelled person (its centre in the top of the
   person's box), with no helmet of any kind labelled, no crowd box, and no drawn or printed person.
2. **3,561 of them were screened**: every validation and test one, and from the training split
   those whose hat is small in the photo (as on CCTV) plus a sample of the rest.
3. **Checked with models and by eye.** Left out:

   | Reason | Photos |
   |---|---|
   | a person the COCO model (yolo26l) may see (0.35-0.6) isn't labelled | 751 |
   | a real helmet labelled "Hat", or unclear: every hat the model calls a helmet (>= 0.25, 1,354 hats in 17 sheets of crops) was looked at (`review_left_out.csv`) | 50 |
   | the model in use is sure of a helmet (>= 0.5) away from every hat: maybe a real one, unlabelled | 46 |
   | the model in use is unsure about a vest (0.25-0.5) | 36 |
   | most labelled people aren't found (a photo stored turned, or wrong boxes) | 5 |
   | no credit, or the photo is stored turned | 2 |

   2,671 were kept. People the COCO model is sure of (>= 0.6) that Open Images didn't box, often
   in the background, were added as labels (652). Vests the model in use is sure of (>= 0.5) were
   labelled (21).

   **A second review** (29 Sep, after the first training): every hat in the chosen photos that
   the model scores 0.05-0.25 as a helmet (585 hats, 8 sheets). 13 photos showed a real helmet
   labelled "Hat" (pith, police, riding, bicycle, military, hard hat) or were unclear, and were
   left out (`review_left_out.csv`, second part).
4. **Splits by photographer.** All photos by one Flickr author are in the same split (70 / 10 / 20 %
   by a hash of their page), so no test photo shows a scene or a person training has seen.
   `prepare_caps_data.py` also leaves out any photo that looks like a ppe4 validation or test photo,
   and any test photo that looks like a training photo.

**Labels:** person and vest only. A hat gets no label, so to the detector it is background: that
is the lesson. Nothing is labelled a helmet, because none of these photos has one.
