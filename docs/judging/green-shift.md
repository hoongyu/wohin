# Wohin and the Green Shift Hacks judging criteria

Judging criterion: **Environmental Impact**, i.e. how the project actually reduces waste, emissions, resource use or environmental harm. Claims below stay qualitative unless a cited source supports them; we have not measured Wohin's effect on real sorting behaviour.

## Environmental Impact

**Where the harm comes from.** Recycling only works if items reach the right stream. The Berlin Senate states that:

- if foreign material in a Wertstofftonne impairs sorting and is not re-sorted, the **whole load must go to residual waste**, and persistent misuse can lead to the bins being withdrawn ([source](https://www.berlin.de/sen/uvk/umwelt/kreislaufwirtschaft/service/privathaushalte/wertstoffsammlung/));
- many Berliners put food leftovers into household waste, or spoiled food together with its packaging into the Biotonne, while calorie-rich kitchen waste produces the most biogas ([source](https://www.berlin.de/sen/uvk/umwelt/kreislaufwirtschaft/projekte/bioabfaelle/biotonne-in-wohnanlagen/was-kommt-in-die-biotonne/)).

**How Wohin addresses it.**

1. **Keeps foreign material out of the Wertstofftonne.** The rule engine routes electrical devices and batteries to take-back points, glass to the glass container, paper to the paper bin, and food to the Biotonne, all of which the Senate lists as not allowed in the Wertstofftonne.
2. **Moves food waste into the Biotonne, without packaging.** Organic items always lead to the Biotonne, with an explicit "unpack it first" outcome and the rule that plastic and "compostable" plastic bags are not allowed while paper bags are.
3. **Glass sorted by colour, and non-packaging glass kept out.** Drinking glasses, mirrors, window glass and light bulbs are kept out of the glass container, as the Senate requires.
4. **Deposit containers go back to the shop** instead of into a bin (25 cents deposit per single-use bottle or can with the DPG logo, [source](https://www.verbraucherzentrale.de/wissen/umwelt-haushalt/abfall/fragen-und-antworten-alles-zum-einwegpfand-und-dosenpfand-11505)).
5. **Electrical devices and batteries go to take-back**, which supermarkets of 800 m² or more that sell electrical devices must offer for free since July 2022 ([source](https://www.umweltbundesamt.de/presse/pressemitteilungen/25000-zusaetzliche-rueckgabestellen-fuer)).
6. **Wearable clothes go to textile collection or donation** rather than the bin ([source](https://www.berlin.de/sen/uvk/umwelt/kreislaufwirtschaft/service/privathaushalte/alttextilien/)).

**Why it reaches the people who need it.** The target users are newcomers who do not read German or know the item names used in the official A–Z lists. Wohin works from a photo or a word in English or German.

**Not adding harm.** The model is a frozen pretrained backbone with a linear head: training took about two minutes on a CPU after a one-time embedding pass of about four minutes, and inference runs on a free CPU instance. No GPU was used.

## Target users

People who recently moved to Berlin (international students, newcomers), and anyone holding an item and unsure where it goes.

## Approach

Photo → material classifier (OpenCLIP ViT-B/32 + calibrated logistic regression, 0.899 test accuracy over 8 classes) → follow-up questions → rule engine built only from official sources → bin, explanation and source link. Details and metrics: `README.md`.

## Limitations

- Expected benefit is qualitative; there is no user study or field measurement yet.
- Accuracy is measured on public datasets (USA, Australia), not on photos taken in Berlin homes.
- Rules that our sources do not cover are not stated as fact; for those the app links to the BSR A–Z.
- Berlin only. Other cities need their own rule file (`rules/<city>.yaml`); the model itself is city-independent.

## Tools used

Python, PyTorch, OpenCLIP, timm, scikit-learn, Streamlit; datasets TrashNet (MIT) and RealWaste (CC BY 4.0 via UCI; the authors' mirror states CC BY-NC-SA 4.0); built with the AI coding assistant Claude Code. Full list and licences: README section 11.
