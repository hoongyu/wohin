# Build log: Wohin MVP

Cloud Claude Code session, branch `mvp`. Session started 2026-11-02 14:54 UTC (15:54 CET), after the Neural Storm window opened at 15:30 CET. Nothing was pushed to `main`, nothing was merged, no PR was opened, nothing was deployed, and the repository visibility was not changed.

## What was built

| Milestone | Status | Where |
|---|---|---|
| M1 Data | done | `wohin/data.py`, `data/manifest.csv` (7,279 rows) |
| M2 Embeddings, baseline, head | done | `wohin/embed.py`, `wohin/train.py`, `artifacts/model/head.joblib` |
| M3 Evaluation (+ optional EfficientNet-B0) | done | `wohin/evaluate.py`, `artifacts/metrics/test.json`, `docs/img/confusion_matrix.png`, `docs/img/reliability.png`, `artifacts/model/head_effnet.joblib` |
| M4 Rules and lexicon | done | `rules/berlin.yaml`, `wohin/rules.py`, `rules/items.yaml` (60 items), `wohin/lexicon.py` |
| M5 App | done | `wohin/predict.py`, `app.py`, screenshots `docs/img/app_*.png`, `docs/img/demo.gif` |
| M6 README and docs | done | `README.md`, `docs/judging/neural-storm.md`, `docs/judging/green-shift.md`, `requirements.txt`, `deploy/huggingface/` |
| M7 Hand-off | done | this file |

Definition of done, checked in this session:

- `uv run pytest`: **34 passed**.
- `uv run python -m wohin.train` from cached embeddings: reproduced the same test metrics on two runs (C, T, threshold and all scores identical).
- `uv run streamlit run app.py`: ran headless and was driven with Playwright/Chromium through the photo flow, low-confidence flow, English search and German search, with no page errors and no Streamlit exceptions.
- README complete except `TODO(owner)` placeholders for team, live link and licence (filled in on 2026-11-03).

Additions beyond the task: a near-duplicate leakage check in the evaluation, a zero-shot confusion matrix, top-3 accuracy, and a demo GIF built from the screenshots.

## Final metrics (test split, 1,093 images; from `artifacts/metrics/test.json`)

| Model | Accuracy | Macro-F1 | Top-3 | ECE | Coverage @ threshold | Accuracy on covered |
|---|---|---|---|---|---|---|
| Zero-shot CLIP ViT-B/32 | 0.676 | 0.618 | 0.873 | 0.118 | 0.327 (t = 0.954) | 0.941 |
| **Linear head, CLIP ViT-B/32 (deployed)** | **0.899** | **0.890** | **0.992** | **0.027** (0.098 before temperature) | **0.834 (t = 0.726)** | **0.951** |
| Linear head, EfficientNet-B0 | 0.868 | 0.861 | 0.986 | 0.026 | 0.794 (t = 0.756) | 0.940 |

- Deployed head: C = 0.003, temperature 0.622, threshold 0.726 (lowest threshold with ≥ 95 % validation accuracy).
- Per source, deployed head: RealWaste acc 0.900 / macro-F1 0.892; TrashNet acc 0.897 / macro-F1 0.874.
- Weakest class: residual (F1 0.736). Main confusion: plastic predicted as glass (12 of 211).
- Abstained test photos: 181; true class in the top three for 95.6 % of them.
- Near-duplicates (CLIP cosine ≥ 0.95 to a train image): 57 test images (5.2 %); accuracy without them 0.896.
- Single-image CPU latency (preprocess + forward, 4 threads): CLIP 55.9 ms, EfficientNet-B0 23.9 ms.

Split sizes: train 5,094 / val 1,092 / test 1,093. Per source and class:

| class | RealWaste train/val/test | TrashNet train/val/test |
|---|---|---|
| paper | 350/75/75 | 416/89/89 |
| cardboard | 323/69/69 | 282/61/60 |
| glass | 294/63/63 | 350/75/76 |
| metal | 553/118/119 | 287/61/62 |
| plastic | 645/138/138 | 337/72/73 |
| organic | 593/127/127 | – |
| textile | 222/48/48 | – |
| residual | 346/75/74 | 96/21/20 |

## Blocked hosts and fallbacks

| Need | Primary | Result | Fallback used |
|---|---|---|---|
| TrashNet | raw.githubusercontent.com zip | **worked** (1.8 s) | – |
| RealWaste | archive.ics.uci.edu | **blocked** (proxy CONNECT 403) | `git clone --depth 1 https://github.com/sam-single/realwaste` (42 s; 4,752 images, counts match UCI) |
| OpenCLIP weights | huggingface.co (`laion2b_s34b_b79k`) | **blocked** (403) | GitHub release `vit_b_32-laion2b_e16-af8dbd0c.pth` (605 MB) |
| EfficientNet-B0 weights | timm via Hugging Face | **blocked** (403) | GitHub release `efficientnet_b0_ra-3dd342df.pth` |
| torch | download.pytorch.org/whl/cpu | **blocked** (403) | default PyPI wheels (CUDA build, ~3 GB of downloads; runs on CPU) |

Also: GitHub web pages and api.github.com returned 403 through the proxy, but `git clone` and release downloads worked.

**Important consequence:** the design named the `laion2b_s34b_b79k` checkpoint, but it is only on Hugging Face. The fallback is a *different* LAION-2B checkpoint (`laion2b_e16`). The whole pipeline (embeddings, head, zero-shot baseline, app) is pinned to `laion2b_e16`, so training and the app match. On a Hugging Face Space, OpenCLIP loads the same `laion2b_e16` weights (from GitHub or its HF mirror `timm/vit_base_patch32_clip_224.laion2b_e16`). Do not switch the app to `laion2b_s34b_b79k` without re-running `wohin.embed` and `wohin.train`.

## Timings (4 CPU cores, 15 GB RAM, no GPU)

| Step | Time |
|---|---|
| `uv sync` (torch with CUDA wheels from PyPI) | ~20 min, dominated by downloads |
| Data download + manifest | TrashNet 1.8 s, RealWaste clone 42 s, manifest 12 s |
| CLIP model load | 25 s |
| **CLIP embedding pass, 7,279 images** | **257 s (28.3 images/s)** |
| EfficientNet-B0 embedding pass | 407 s (17.9 images/s), slowed down because training and the app ran at the same time |
| `wohin.train` (C search + calibration + evaluation, both backbones) | 2 min 3 s |
| App: first photo prediction (cold) | ~8 s incl. model load; later photos ~1 s |

## Known issues

1. **Lid colours.** The bin card shows a colour swatch for every bin. Our sources state the Wertstofftonne lid colour (yellow or orange) and the glass colours, but not the paper bin, Biotonne or household-waste lid colours. Those swatches (blue, brown, dark grey) follow common practice but the text does not name a colour (`colour_verified: false` in `rules/berlin.yaml`). Owner: check and either confirm with a source or keep as is.
2. **Domain gap.** No test on photos taken in Berlin households; dataset photos are single items on a white background (TrashNet) or landfill (RealWaste).
3. **Deployment.** download.pytorch.org was blocked here, so the Space build could not be tested in this session. Checked afterwards on the owner's laptop, see "Local Docker check" below. Hugging Face retired its built-in Streamlit SDK on 2025-04-30 and a Space without a Dockerfile defaults to Python 3.10, which scikit-learn 1.9.1 and Streamlit 1.66.1 do not support, so the Space now builds from `deploy/huggingface/Dockerfile`.
4. **scikit-learn pinned to 1.9.1** in `requirements.txt`, because `head.joblib` is a pickled scikit-learn pipeline.
5. **Session state:** the photo and search tabs keep separate state. Uploading a new photo resets the photo flow; changing the search query resets the search flow.
6. **Camera input** sits behind a "Use camera" toggle. It needs HTTPS or localhost to get browser camera permission (fine on HF Spaces).
7. **Item coverage:** 60 lexicon items; 9 are `verified: false` and only link to the BSR A–Z with a hint that repeats verified facts.
8. **The battery outcome is `partial`:** the Senate page bans batteries from the Wertstofftonne, but the take-back advice is not in our sources.
9. **RealWaste licence:** we used the authors' GitHub mirror (README: CC BY-NC-SA 4.0) because UCI (CC BY 4.0) was blocked. No images are redistributed. Screenshots only show TrashNet (MIT) images.

## Local Docker check (2026-11-02, after the cloud session)

Run by the owner's local Claude Code session on the owner's Windows 11 laptop (Docker 29.5.2), on the folder that `deploy/huggingface/make_space.py` writes, which is what the Space receives:

- `uv sync` and `uv run pytest -q` on branch `mvp`: 34 passed in 32.6 s.
- `docker build` finished in 1,160 s: `pip install -r requirements.txt` 249 s (torch 2.15.0+cpu and torchvision 0.29.2+cpu from the CPU index), weight download 79 s, unpacking the 3.57 GB image 414 s.
- `docker run -p 18501:8501`: `/_stcore/health` returned `ok`, `GET /` returned HTTP 200.
- In a container started with `--network none`, the bottle photo cropped from `docs/img/app_photo_prediction.png` gave glass 0.999 (confident), as in that screenshot. First prediction 38.3 s including model load, then 0.09 s.

## Hosting change (2026-11-03)

- Creating the Hugging Face Docker Space from the owner's free account failed with `402 Payment Required`: Gradio and Docker Spaces on the free CPU tier now need a PRO subscription (USD 9 per month, https://huggingface.co/pricing). Nothing was created.
- The live demo moves to Streamlit Community Cloud, which is free (up to about 2.7 GB of memory). Community Cloud installs the first dependency file it finds, next to the entry point before the repository root, and the root `uv.lock` pins the CUDA build of torch on Linux. So `deploy/streamlit/` holds an entry point that runs `app.py` and its own CPU-only `requirements.txt`.
- `uv pip compile deploy/streamlit/requirements.txt` for Python 3.11 on Linux x86_64 (Community Cloud tries uv first, then pip): 60 packages, torch 2.15.0+cpu, torchvision 0.29.2+cpu, no CUDA packages.
- New `tests/test_deploy.py`: the two requirements files match, and the entry point runs the app under Streamlit's AppTest (the text search for "pizza box" gives Cardboard). `uv run pytest -q`: 36 passed.

## What the owner still has to do

- [x] Team name and members in `README.md`: Lampwright, hoongyu (solo).
- [x] Licence: MIT for the code (`LICENSE`); the trained heads in `artifacts/model/` are CC BY-NC-SA 4.0 (`README.md` section 12).
- [ ] Review the rules and texts in `rules/berlin.yaml` and `rules/items.yaml`, especially known issue 1 (lid colours).
- [ ] Deploy on Streamlit Community Cloud following `deploy/streamlit/DEPLOY.md` (a Hugging Face Docker Space now needs PRO, see "Hosting change"); check the link in `README.md` ("Live demo") and in Devpost.
- [ ] Try the deployed app with a few real photos from home; consider adding them as a qualitative example (do not add to the metrics).
- [ ] Optionally record a narrated demo video (a GIF made from screenshots is at `docs/img/demo.gif`).
- [ ] Merge `mvp` into `main` (or open a PR yourself), then make the repository public.
- [ ] Submit on Devpost (text ready in `docs/submission/neural-storm.md`): Neural Storm before **2026-11-04 15:30 CET**; Green Shift Hacks before **2026-11-17 23:00 CET** (use `docs/judging/green-shift.md` for the story).
