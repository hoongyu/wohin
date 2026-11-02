# Wohin: design

Date: 2026-11-02. Status: approved by the owner in chat; written before the Neural Storm build window opened (2026-11-02 15:30 CET). No code exists yet.

## 1. Summary

Wohin ("where does it go?") is a photo-based waste sorting assistant for Berlin. You photograph one item. An image classifier predicts its material. A rule engine built from official Berlin sources turns the material, plus one or two follow-up answers, into the right bin, and explains why with a link to the source.

Built for Neural Storm 2026, track Green Planet, problem statement **GP-05: Smart Waste Segregation System**, inside the official 48-hour window. The same project is then submitted unchanged to Green Shift Hacks.

## 2. Problem

Material alone does not decide the bin:

- A glass bottle goes to the glass container, sorted by colour (white, brown, green). A drinking glass, mirror or window pane does not go there at all.
- A plastic bottle with the deposit logo goes back to the shop for 25 cents. Without it, it goes into the Wertstofftonne.
- Paper goes into the paper bin, but used kitchen paper goes into the Biotonne.
- Food goes into the Biotonne, but never in a plastic bag, including "compostable" plastic bags.

People new to Berlin (international students, newcomers) do not know these rules, and the official A–Z lists assume you know the German name of the item.

Mistakes have a cost. The Berlin Senate states that when a Wertstofftonne contains foreign material that impairs sorting, the collector can demand re-sorting, and if that does not happen, the whole load goes to residual waste. The Senate also states that many Berliners put food waste into household waste, or spoiled food together with its packaging into the Biotonne, while kitchen waste is exactly what produces the most biogas. Sources are listed in `docs/cloud-task.md`, appendix A.

## 3. Users

- Primary: people who moved to Berlin recently and are unsure about German waste rules or labels.
- Secondary: anyone holding an item and unsure where it goes.

## 4. Scope

In:

1. Single-item photo classification into 8 material classes.
2. Berlin rule engine with follow-up questions and a source link on every answer.
3. German and English UI.
4. Text search fallback ("Pizzakarton", "receipt", "tea bag").
5. An evaluation report with real metrics and a baseline.
6. README that meets the Neural Storm checklist, screenshots and a demo GIF.

Out: detecting several items in one photo, accounts, a mobile app, gamification, container maps (we link to official services instead).

## 5. Approach (chosen: frozen backbone + linear head)

Considered:

1. **Frozen pretrained backbone, train a linear head, add a rule engine with follow-up questions. Chosen.** Trains in minutes on CPU, works in a CPU-only cloud session, gives clean metrics.
2. Full fine-tuning of a CNN. Maybe a few points more accuracy, but there is no GPU available; too risky in 48 hours.
3. Ask a vision LLM directly. Broadest coverage, but needs a paid API key and gives no measurable model accuracy, which is 35% of the Neural Storm score.

## 6. Architecture

```
photo -> preprocess -> frozen backbone -> embedding -> linear head -> 8 material probabilities
                                                                         |
            bin + explanation + source  <-  rule engine (rules/berlin.yaml)  <-  follow-up answers
```

Modules, each with one job and a small interface:

| Module | Job | Interface |
|---|---|---|
| `wohin/data.py` | Download TrashNet and RealWaste, map labels to the 8 unified classes, make seeded stratified splits | writes `data/manifest.csv` (path, source, original_label, label, split) |
| `wohin/embed.py` | Load a backbone with download fallbacks, extract and cache embeddings | `embed(paths) -> np.ndarray`, cache in `artifacts/embeddings/` |
| `wohin/train.py` | Fit the linear head, tune on validation, calibrate | writes `artifacts/model/head.joblib`, `artifacts/metrics/train.json` |
| `wohin/evaluate.py` | Test metrics, baseline, plots | writes `artifacts/metrics/test.json` and PNGs under `docs/img/` |
| `wohin/rules.py` + `rules/berlin.yaml` | Material + answers -> outcome | `next_question(material, answers)`, `outcome(material, answers)` |
| `wohin/lexicon.py` + `rules/items.yaml` | Text search over item names (DE/EN) | `search(text) -> [(item, material, preset_answers)]` |
| `wohin/predict.py` | Image -> top-k materials with calibrated probabilities | `predict(image) -> [(label, prob)]` |
| `app.py` | Streamlit UI | `streamlit run app.py` |

## 7. Data

- **TrashNet** (github.com/garythung/trashnet, MIT): 2,527 images, 6 classes (glass, paper, cardboard, plastic, metal, trash). `data/dataset-resized.zip`, 42.8 MB.
- **RealWaste** (UCI dataset 908, CC BY 4.0): 4,752 images at 524×524, 9 classes: Cardboard 461, Food Organics 411, Glass 420, Metal 790, Miscellaneous Trash 495, Paper 500, Plastic 921, Textile Trash 318, Vegetation 436. Also mirrored in the authors' repository github.com/sam-single/realwaste (about 656 MB).
- **Unified classes (8):** paper, cardboard, glass, metal, plastic, organic (Food Organics + Vegetation), textile (Textile Trash), residual (trash + Miscellaneous Trash).
- Splits: 70/15/15, stratified by unified label within each source dataset, seed 42.
- Raw images are not committed (size). The download script, the manifest and the trained head are committed.

Known issues to handle and report:

- Domain gap: TrashNet photos are on a white background, RealWaste photos are from a landfill. Report metrics per source dataset, so we can see whether the model learned materials or backgrounds.
- Class imbalance (TrashNet has only 137 "trash" images): use balanced class weights; report per-class scores.
- Near-duplicate photos of the same object may sit in different splits and inflate scores. State this as a limitation.

## 8. Model

- Backbone: OpenCLIP ViT-B/32 (LAION-2B). One backbone serves both the linear probe and the zero-shot baseline, which makes the comparison clean. Lighter alternative if time permits: timm EfficientNet-B0; pick the deployed model on accuracy versus latency.
- Head: multinomial logistic regression (scikit-learn), `class_weight="balanced"`, C tuned on the validation split.
- Calibration: temperature scaling on validation. Choose an abstain threshold on validation; below it, the app asks the user to confirm the material from the top three instead of guessing. Report coverage versus accuracy.
- Baseline: CLIP zero-shot with a small prompt ensemble ("a photo of {class} waste", ...).
- Metrics on the test split: accuracy, macro-F1, per-class precision/recall/F1, confusion matrix, per-source breakdown, expected calibration error, coverage at the chosen threshold. Report what we get; no target number is promised.

## 9. Rule engine

`rules/berlin.yaml` holds bins, questions and per-material decision flows. Every outcome carries: bin id, labels and explanation in DE and EN, the bin colour, a source URL, and `verified: true|partial|false`.

- `verified: true` outcomes are stated plainly.
- Anything else shows the best guidance plus "please check the BSR A–Z" with a link. The app never presents an unverified rule as fact.

Flows (the verified facts behind them are in `docs/cloud-task.md`, appendix A):

- **glass**: packaging (bottle or jar)? no -> household waste (drinking glass, mirror, window glass) or, for light bulbs, special handling. Deposit logo? yes -> return to shop. Colour? white -> white glass; brown -> brown glass; green, other colours or unsure -> green glass. Metal lids -> Wertstofftonne.
- **plastic / metal**: deposit logo on a bottle or can? yes -> return to shop. Contains electronics or batteries? yes -> shop take-back or BSR recycling centre. Otherwise -> Wertstofftonne (packaging and non-packaging items of the same material).
- **paper / cardboard**: used kitchen paper? yes -> Biotonne. Beverage carton? yes -> Wertstofftonne. Otherwise -> paper bin.
- **organic**: -> Biotonne, with a note: no plastic or compostable plastic bags; paper bags are fine; remove packaging first.
- **textile**: wearable and clean? yes -> textile collection (donation container); worn out or dirty -> household waste (`partial`).
- **residual**: hygiene products, nappies, vacuum bags, cat litter, ash -> household waste. Batteries or electronics -> take-back.

## 10. UI

- Streamlit, language toggle DE/EN.
- Input: camera, upload, or text search.
- Output: top three materials with calibrated probabilities, the bin card (colour plus text label, never colour alone), explanation, source link, follow-up questions as buttons with an "I'm not sure" option.
- Footer: rules as of 2026-11-02 from official sources; not affiliated with BSR, ALBA or the Senate.

## 11. Testing

pytest:

- Every outcome in `rules/berlin.yaml` is reachable and has a source URL.
- Every question has DE and EN text.
- Lexicon lookups return the expected material.
- No image appears in two splits.
- Prediction smoke test on a tiny fake embedding, so tests run without downloading weights.

## 12. Deliverables

Neural Storm README (English): title and problem statement number, team name and members (filled in by the owner), problem and solution, tech stack, setup, running locally, screenshots or GIF, model metrics, AI tool disclosure (Claude Code, pretrained models, datasets), and a note that planning docs predate the window while all code was written after 2026-11-02 15:30 CET.

Also: judging-criteria mappings for Neural Storm and Green Shift under `docs/judging/`, and deployment files for Hugging Face Spaces. The owner deploys, makes the repository public and submits.

## 13. Risks

| Risk | Mitigation |
|---|---|
| Cloud network blocks Hugging Face or UCI | GitHub-hosted fallbacks for weights and RealWaste (listed in `docs/cloud-task.md`) |
| CPU only | Frozen backbone; embeddings cached once |
| Domain gap between datasets | Per-source metrics, abstain threshold, honest limitations |
| A wrong rule misleads users | Only verified rules stated as fact; others link to BSR |
| Licences | TrashNet MIT, RealWaste CC BY 4.0 with attribution, OpenCLIP MIT, timm Apache-2.0 |
