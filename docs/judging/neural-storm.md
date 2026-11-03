# Wohin and the Neural Storm 2026 judging criteria

Track 2 *Green Planet*, problem statement **GP-05: Smart Waste Segregation System**. We narrow GP-05 to a concrete, local version: segregation at the source, by the person holding the item, under the real rules of one city (Berlin).

## Technical Implementation (35 %)

*Code quality, AI/ML techniques, model accuracy, technical depth.*

- **AI/ML pipeline in Python:** frozen OpenCLIP ViT-B/32 → cached embeddings → logistic regression with balanced class weights and `C` tuned on validation → temperature scaling → abstain threshold chosen on validation. Code: `wohin/embed.py`, `wohin/train.py`, `wohin/metrics.py`.
- **Measured against a baseline:** a CLIP zero-shot baseline on the same backbone, with a prompt ensemble. The trained head reaches **0.899 accuracy / 0.890 macro-F1** on 1,093 test images vs. **0.676 / 0.618** for zero-shot. Evidence: `artifacts/metrics/test.json`, README section 5.
- **Calibrated uncertainty, used in the product:** ECE drops from 0.098 to 0.027 after temperature scaling. At the validation-chosen threshold the app answers 83 % of test photos with 95 % accuracy and asks the user for the rest (true class in the top three 96 % of the time). Evidence: `docs/img/reliability.png`, `test.json` → `selective`.
- **Second backbone compared:** EfficientNet-B0 gets 0.868 accuracy at 24 ms/image vs. CLIP 0.899 at 56 ms/image on CPU; the trade-off is documented and CLIP is deployed.
- **Robustness checks:** per-source metrics (RealWaste 0.900, TrashNet 0.897 accuracy), per-class precision/recall/F1, confusion matrices, and a near-duplicate leakage check (5.2 % of test images have a near-duplicate in train; accuracy without them is 0.896).
- **Engineering:** small modules with one job each, a declarative rule engine (`rules/berlin.yaml`) instead of hard-coded if-chains, download fallbacks for blocked hosts, seeded splits, and 36 pytest tests (rules reachability, sources, DE/EN completeness, lexicon lookups, split disjointness, a prediction smoke test that needs no weights, the deployment entry point). `uv run python -m wohin.train` reproduces every number from cached embeddings.

## Impact & Usefulness (25 %)

- **Real users with a real problem:** newcomers to Berlin who do not know German waste rules or item names. The app works in English and German and accepts a photo or a word in either language.
- **Solves the part material alone cannot:** follow-up questions capture the deciding facts (deposit logo, glass colour, kitchen paper, beverage carton, still packaged, wearable) and the rule engine maps them to the right bin, including the deposit return that pays 25 cents.
- **Trustworthy by design:** every answer cites its source; anything not covered by an official source is marked and points to the BSR A–Z instead of guessing. Low-confidence photos are not guessed either.
- **Why it matters:** the Senate states that a contaminated Wertstofftonne load can end up entirely in residual waste, and that food waste wrongly put into household waste is lost for biogas. Both are sorting mistakes at the source, which is exactly where Wohin helps. Details: `docs/judging/green-shift.md`.

## README & Documentation (20 %)

The README covers every required item: title and problem statement number, team (Lampwright: hoongyu, solo), problem and solution, technology stack, setup, running locally, screenshots and a demo GIF, model metrics with a baseline, limitations, and disclosure of Claude Code, pretrained models and datasets. Supporting docs: the design (`docs/superpowers/specs/2026-11-02-wohin-design.md`), this file, `docs/judging/green-shift.md`, `docs/build-log.md`, and `deploy/streamlit/DEPLOY.md`.

## Demo & Presentation (20 %)

- **Working demo:** `uv run streamlit run app.py`. Screenshots in `docs/img/` were taken automatically from the running app (photo → prediction → questions → bin card; low-confidence path; English and German text search).
- **Deployment:** the live demo runs on Streamlit Community Cloud from `deploy/streamlit/` (an entry point and CPU-only requirements, checked by `tests/test_deploy.py`): https://wohin-berlin.streamlit.app. It was checked in a browser on 2026-11-03 with the text search and three test photos, all classified correctly (`docs/build-log.md`, "Live deployment"). A Hugging Face Docker Space is also ready in `deploy/huggingface/` for PRO accounts; that image was built and run locally, and it predicts offline because the weights are baked in (`docs/build-log.md`, "Local Docker check").
- **Demo GIF:** `docs/img/demo.gif`.
