# Devpost text: Neural Storm 2026

Paste each section into the matching Devpost field; the sections from Inspiration to What's next go together into "About the project". Track: Green Planet. Problem statement: **GP-05: Smart Waste Segregation System**. Every figure below is from `README.md` or `docs/build-log.md`, which take them from `artifacts/metrics/test.json`.

## Elevator pitch

Photograph one item and Wohin names the right Berlin bin: a calibrated image classifier, one or two follow-up questions, and rules taken from official sources.

## Inspiration

In Berlin, the material does not decide the bin. A glass bottle goes to the glass container, but a drinking glass must not. A plastic bottle with the deposit logo goes back to the shop. Used kitchen paper goes into the Biotonne, not the paper bin. The official A–Z lists assume you already know the German name of the item, so newcomers guess, and the Berlin Senate says that a Wertstofftonne with too much foreign material can end up as residual waste.

## What it does

Photograph one item. Wohin predicts its material (8 classes) with calibrated probabilities, asks one or two follow-up questions ("Does it have the deposit logo?", "What colour is the glass?"), and shows the right Berlin bin with an explanation and a link to the official source. When the model is unsure, it does not guess: you pick the material from its top three. A German and English text search covers about 60 common items.

## How we built it

- **Data:** TrashNet (2,527 images) and RealWaste (4,752 images), mapped to 8 classes and split 70/15/15 by class within each dataset (seed 42).
- **Model:** frozen OpenCLIP ViT-B/32 (LAION-2B) embeddings, a logistic-regression head tuned on validation macro-F1, temperature scaling, and an abstain threshold set to the lowest value that reaches 95% validation accuracy.
- **Baselines:** CLIP zero-shot on the same backbone, and an EfficientNet-B0 head for accuracy versus speed.
- **Rules:** a small decision tree per material in YAML, built only from official Berlin sources. Every outcome carries a source URL and a verification level, and only verified rules are stated as fact.
- **App:** Streamlit, German and English; 36 pytest tests.

## Challenges we ran into

- **Two very different datasets** (white background versus landfill). We report accuracy per dataset to check that the model learns materials, not backgrounds.
- **No GPU.** A frozen backbone and cached embeddings kept it practical: the CLIP embedding pass took 257 s and training both heads took about 2 minutes on 4 CPU cores.
- **Blocked downloads** in the build environment. The data and weights came from GitHub copies, so the pipeline is pinned to the checkpoint we could actually get.
- **Material is not enough.** The same material can belong in different bins, so the app asks follow-up questions instead of guessing.

## Accomplishments that we're proud of

- Test accuracy **0.899** and macro-F1 **0.890** on 1,093 test images, against 0.676 and 0.618 for zero-shot CLIP on the same backbone.
- Calibrated probabilities: expected calibration error 0.027 (0.098 before temperature scaling).
- The app answers directly for 83% of test photos with 95% accuracy; for the rest, the true material is among the top three in 96% of cases.
- Similar accuracy on both datasets: 0.900 on RealWaste, 0.897 on TrashNet.

## What we learned

- For a tool people should trust, calibration and an honest "not sure" matter as much as raw accuracy.
- A near-duplicate check is cheap and makes the numbers more credible: 5.2% of test images have a near-copy in training, and accuracy without them is 0.896.
- Rules need sources: 9 of the 60 searchable items could not be verified, so the app links to the official BSR A–Z for those instead of inventing an answer.

## What's next for Wohin

- Measure accuracy on photos taken in Berlin homes, which look different from both datasets.
- Confirm the remaining bin lid colours from an official source.
- More items in the text search, and a layout tuned for phones.

## Built with

python, pytorch, open-clip, timm, scikit-learn, scipy, numpy, pandas, pillow, pyyaml, matplotlib, streamlit, pytest, uv, playwright

## Links

- Repository: https://github.com/hoongyu/wohin
- Live demo: https://wohin-berlin.streamlit.app
- Demo GIF (no video): https://github.com/hoongyu/wohin/blob/main/docs/img/demo.gif

## Images

- Thumbnail (3:2): `docs/img/devpost_thumbnail.png`
- Gallery: `docs/img/app_photo_prediction.png`, `docs/img/app_photo_result.png`, `docs/img/app_low_confidence.png`, `docs/img/app_search_teabag.png`, `docs/img/app_search_pizzakarton_de.png`, `docs/img/demo.gif`

## Team

Lampwright: hoongyu (solo), master's student at TU Berlin.

## AI disclosure

Built with Claude Code (Anthropic), which wrote the code, ran the training and evaluation, and drafted the documentation under the owner's design and direction. The design and task documents were written before the build window opened; all code, models and metrics were created after 2026-11-02 15:30 CET (see the git history and README section 11).
