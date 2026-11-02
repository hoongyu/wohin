# Cloud task: build the Wohin MVP

You are a cloud Claude Code session working in this repository (hoongyu/wohin, private). Read this file first, then the design: `docs/superpowers/specs/2026-11-02-wohin-design.md`. The design is approved; build it. If the two disagree, this file wins.

## 1. Deadlines (all times CET = UTC+1; Berlin has been on winter time since 2026-10-25)

| What | When |
|---|---|
| Neural Storm build window opens | 2026-11-02 15:30 CET (20:00 IST) |
| Neural Storm submission deadline | 2026-11-04 15:30 CET (official: November 4, 2026, 8:00 PM IST) |
| Green Shift Hacks deadline | 2026-11-17 23:00 CET |

Your session starts after the window opened. Finish everything in this file in one run if you can; the owner reviews and polishes on 11-03.

## 2. Hard rules

1. **All code, data processing and assets are created now, inside the window.** Only `docs/cloud-task.md` and the design existed before 15:30 CET. Do not import code from other projects.
2. Work on a new branch `mvp`. Commit early and often with clear messages. Push the branch. Do not push to `main`, do not merge, do not open a pull request.
3. Do not make the repository public, do not deploy anything, do not create accounts. The owner does those.
4. No paid API keys. Never commit secrets, `.env` files, raw datasets or backbone weights.
5. Python is the primary language (Neural Storm requires it for the AI/ML part). Use `uv` with `pyproject.toml`, Python 3.11.
6. README and code comments in English.
7. Never invent facts. Berlin rules must come from appendix A; anything not covered there is marked `verified: false` and the app links to the BSR A–Z instead of asserting. Metrics in the README must be the numbers your scripts actually produced.
8. Leave the team name and member names as a visible placeholder for the owner: `TODO(owner): team name and members`.

## 3. Environment and network

The session is CPU-only. The network may be restricted to an allowlist (PyPI, GitHub and Google Cloud storage are allowed; Hugging Face, download.pytorch.org and archive.ics.uci.edu may be blocked). Try the primary source; on failure use the fallback and record what was blocked in `docs/build-log.md`.

| Need | Primary | Fallback |
|---|---|---|
| TrashNet | `https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip` (42.8 MB) | `git clone --depth 1 https://github.com/garythung/trashnet` |
| RealWaste | `https://archive.ics.uci.edu/static/public/908/realwaste.zip` (656.6 MB, CC BY 4.0) | `git clone --depth 1 https://github.com/sam-single/realwaste` (authors' mirror, same 4,752 images; its README states CC BY-NC-SA 4.0) |
| OpenCLIP ViT-B/32 | `open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")` (Hugging Face) | download `https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-laion2b_e16-af8dbd0c.pth` (605 MB) and pass the local path as `pretrained` |
| EfficientNet-B0 (optional) | `timm.create_model("efficientnet_b0", pretrained=True)` | `https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0_ra-3dd342df.pth` (21 MB), load the state dict |
| torch | CPU wheels from `https://download.pytorch.org/whl/cpu` | default PyPI wheels (large, but they work) |

Licence notes for the README: TrashNet MIT; RealWaste distributed by UCI under CC BY 4.0 (the GitHub mirror states CC BY-NC-SA 4.0; this project is non-commercial, attributes the authors and does not redistribute images); OpenCLIP code MIT, weights trained on LAION-2B; timm Apache-2.0.

## 4. Milestones

Do them in order. Each ends with a commit.

**M1 Data.** `wohin/data.py` downloads both datasets into `data/raw/` (gitignored), maps labels to the 8 unified classes from the design, writes `data/manifest.csv` with seeded stratified 70/15/15 splits per source. Print class counts per source and split. Test: no path appears in two splits.

**M2 Embeddings, baseline, head.** `wohin/embed.py` extracts OpenCLIP ViT-B/32 image embeddings for every manifest row and caches them in `artifacts/embeddings/` (gitignored). `wohin/evaluate.py` computes the zero-shot baseline. `wohin/train.py` fits the logistic-regression head (balanced class weights, C tuned on validation), fits temperature scaling on validation, chooses the abstain threshold on validation, and saves `artifacts/model/head.joblib` (commit this, it is small). Time the embedding pass and note it in the build log.

**M3 Evaluation.** On the test split, for baseline and head: accuracy, macro-F1, per-class precision/recall/F1, per-source breakdown, expected calibration error, coverage and accuracy at the threshold. Save `artifacts/metrics/test.json` and `docs/img/confusion_matrix.png`, `docs/img/reliability.png`. If time allows, repeat with EfficientNet-B0 and report the trade-off.

**M4 Rules and lexicon.** `rules/berlin.yaml` and `wohin/rules.py` implement the flows in design section 9 using only appendix A. `rules/items.yaml` and `wohin/lexicon.py`: about 50 common items in DE and EN (Pizzakarton, Kassenbon, Teebeutel, Joghurtbecher, Kronkorken, Alufolie, Tetra Pak, Batterie, Glühbirne, ...), each with material and preset answers; items whose rule is not in appendix A get `verified: false`. Tests: every outcome reachable, every outcome has a source URL, every question has DE and EN text.

**M5 App.** `wohin/predict.py` and `app.py` (Streamlit) as in design section 10: camera or upload or text search, top three with calibrated probabilities, follow-up questions with "I'm not sure", bin card with colour and text, explanation, source link, DE/EN toggle, low-confidence path that asks the user to pick from the top three. It must run with only the committed head plus the backbone download. Take screenshots into `docs/img/` (use headless Chromium via Playwright if available; otherwise leave a note in the build log for the owner).

**M6 README and docs.**

- `README.md` covering every Neural Storm item in section 5, with a Results section built from `artifacts/metrics/test.json`.
- `docs/judging/neural-storm.md` and `docs/judging/green-shift.md`: one section per judging criterion, saying concretely how Wohin addresses it and where the evidence is.
- Deployment files for a Hugging Face Space (Streamlit SDK): `requirements.txt` and the Space metadata block. The owner deploys.

**M7 Hand-off.** `docs/build-log.md`: what was built, the final metrics, blocked hosts and fallbacks used, timings, known issues, and a checklist of what the owner still has to do. Push the branch.

Definition of done: `uv run pytest` passes; `uv run python -m wohin.train` reproduces the metrics from cached embeddings; `uv run streamlit run app.py` works; the README is complete except the team placeholder and the live link; branch `mvp` is pushed.

## 5. Neural Storm 2026 (primary target)

- Organiser: JNTUH, on Devpost (https://neuralstorm.devpost.com/). Students, worldwide, solo allowed.
- Track 2, Green Planet. Problem statement **GP-05: Smart Waste Segregation System**. The 15 problem statements are starting points; narrowing or reinterpreting within the track is allowed.
- Build expectations: a clear problem in the track; AI/ML as a meaningful component; Python as the primary language for AI/ML; a working prototype or MVP; explain the technical approach and show meaningful results; developed during the 48-hour period. Open-source frameworks, pretrained models and AI coding assistants are allowed but must be disclosed in the README.
- Required: a public GitHub repository with the complete source and a README containing:
  1. Project title and problem statement number
  2. Team name and member names
  3. Problem description and proposed solution
  4. Technology stack and libraries used
  5. Setup and installation instructions
  6. Instructions for running the project locally
  7. Screenshots or a demo GIF of the working project
  8. Model accuracy or performance metrics
  9. Disclosure of any AI development tools used (this session: Claude Code)
- Strongly recommended: a live deployment link (Streamlit Cloud, Hugging Face Spaces or Render).
- One submission per team; it can be updated until the deadline.
- Judging: Technical Implementation 35% (code quality, AI/ML techniques, model accuracy, technical depth); Impact & Usefulness 25%; README & Documentation 20% (setup, problem statement, approach, results, screenshots); Demo & Presentation 20% (working demo, deployment link, presentation quality). Ties go to the higher technical score, then the earlier submission.

## 6. Green Shift Hacks (reuse, same project)

- Devpost, students worldwide, teams up to 4. Work must be original and created after 2026-10-19; open-source tools and datasets are fine with credit.
- Judging: Environmental Impact (how the project actually reduces waste, emissions, resource use or environmental harm).
- Submission (the owner posts it): a Devpost page with the story, all team members, a demo video or walkthrough, source code or build instructions; explain the problem, target users, approach, expected environmental benefit, tools used and limitations.
- For `docs/judging/green-shift.md`, keep impact claims qualitative unless appendix A gives a source.

## 7. What the owner does (do not do these)

Fill in the team name and members, deploy the Space, make the repository public, record the demo GIF or video if you could not, and submit on Devpost.

## Appendix A. Berlin rules verified on 2026-11-02

Use these and nothing else as `verified: true`.

**Wertstofftonne (yellow or orange lid)**, source: https://www.berlin.de/sen/uvk/umwelt/kreislaufwirtschaft/service/privathaushalte/wertstoffsammlung/

- Since 2013 Berlin collects used packaging made of metal, plastic and composites **together with non-packaging items of the same materials** ("stoffgleiche Nichtverpackungen").
- Not allowed: electrical devices, batteries, glass containers, paper and cardboard, textiles, wood, food waste, garden waste.
- If foreign material impairs sorting, the collector (ALBA) can demand re-sorting; if that does not happen, the whole load must go to residual waste; persistent misuse can lead to the bins being withdrawn.

**Glass**, same source:

- Packaging glass only, sorted by colour into white, green and brown.
- Not allowed: glass that is not packaging, such as window glass, mirrors, glass tableware, light bulbs.
- Other colours (blue, red, unclear): put into green glass. This is a recommendation from consumer and recycling guidance, not stated on the Senate page; mark it `verified: partial`, source https://www.test.de/Recycling-Buntglas-entsorgen-4780597-0/

**Paper, cardboard, card (paper bin)**, same source: printed paper and packaging made of paper, cardboard and card.

**Biotonne**, source (list quoted from BSR): https://www.berlin.de/sen/uvk/umwelt/kreislaufwirtschaft/projekte/bioabfaelle/biotonne-in-wohnanlagen/was-kommt-in-die-biotonne/

- Allowed: food leftovers, cooked ones too (wrapped in newspaper); old food without packaging; fruit and vegetable scraps, peel (citrus too), eggshells; coffee grounds and filters, tea, tea bags; kitchen paper; green and shrub cuttings, leaves, flowers.
- Not allowed: plastics, packaging, films, bags; metals, aluminium foil, cans; hygiene products, nappies, vacuum cleaner bags; textiles, leather; cat and small-animal litter; treated wood, ash; soil, sand, gravel, stones.
- Paper bags filled with kitchen waste are allowed. Bags made of biodegradable ("compostable") plastic are not.
- The same page says many Berliners wrongly put food leftovers into household waste, or spoiled food with its packaging into the Biotonne, and that calorie-rich kitchen waste produces the most biogas.

**Deposit (Einwegpfand)**, source: https://www.verbraucherzentrale.de/wissen/umwelt-haushalt/abfall/fragen-und-antworten-alles-zum-einwegpfand-und-dosenpfand-11505

- Single-use bottles and cans with the DPG deposit logo carry 25 cents deposit. Shops must take back containers of the same material regardless of brand, shape or where they were bought.

**Small electrical devices**, source: https://www.umweltbundesamt.de/presse/pressemitteilungen/25000-zusaetzliche-rueckgabestellen-fuer

- Since July 2022, food retailers with at least 800 m² of sales area that sell electrical devices must take back old small devices (up to 25 cm edge length) free of charge. BSR recycling centres also accept them.
- Batteries: never in household waste; return them to shops that sell batteries or to BSR recycling centres. Mark `verified: partial` (not covered by the sources above).

**Textiles**, source: https://www.berlin.de/sen/uvk/umwelt/kreislaufwirtschaft/service/privathaushalte/alttextilien/

- Wearable clothes and shoes: textile collection containers or donation.
- Worn-out, dirty or contaminated textiles: household waste, according to the Senate's answers to written parliamentary questions in 2025; media reports contradict this. Mark `verified: partial`.

**Household waste (Hausmüll)**: items the lists above exclude from every recycling stream (hygiene products, nappies, vacuum bags, cat litter, cold ash, glass tableware, mirrors) default to household waste. Mark `verified: partial` unless a source above says so directly.

For everything else, link the BSR A–Z: https://www.bsr.de/abfall-abc
