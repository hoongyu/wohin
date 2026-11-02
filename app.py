"""Wohin: where does it go? Streamlit app.

Run: uv run streamlit run app.py
"""

from __future__ import annotations

import hashlib
import html

import streamlit as st
from PIL import Image

from wohin.lexicon import search
from wohin.predict import is_confident, load_head, predict
from wohin.rules import get_outcome, next_question, outcome

st.set_page_config(page_title="Wohin – Berlin waste sorting", page_icon="♻️", layout="centered")

MATERIAL_NAMES = {
    "paper": {"de": "Papier", "en": "Paper"},
    "cardboard": {"de": "Pappe / Karton", "en": "Cardboard"},
    "glass": {"de": "Glas", "en": "Glass"},
    "metal": {"de": "Metall", "en": "Metal"},
    "plastic": {"de": "Kunststoff", "en": "Plastic"},
    "organic": {"de": "Bioabfall", "en": "Organic"},
    "textile": {"de": "Textil", "en": "Textile"},
    "residual": {"de": "Restmüll / Sonstiges", "en": "Residual / other"},
}

T = {
    "tagline": {"de": "Wohin damit? Foto machen, Fragen beantworten, richtige Tonne finden – nach den Berliner Regeln.",
                "en": "Where does it go? Take a photo, answer a question or two, find the right bin under Berlin's rules."},
    "tab_photo": {"de": "📷 Foto", "en": "📷 Photo"},
    "tab_text": {"de": "🔎 Suche", "en": "🔎 Search"},
    "camera": {"de": "Foto aufnehmen", "en": "Take a photo"},
    "upload": {"de": "…oder Bild hochladen", "en": "…or upload an image"},
    "one_item": {"de": "Ein Gegenstand pro Foto, möglichst formatfüllend.", "en": "One item per photo, filling most of the frame."},
    "top3": {"de": "Das Modell sieht:", "en": "The model sees:"},
    "confident": {"de": "Erkannt: **{m}**. Nicht richtig? Wähle unten das Material.", "en": "Detected: **{m}**. Not right? Pick the material below."},
    "unsure": {"de": "Ich bin mir nicht sicher genug. Welches Material ist es?", "en": "I'm not confident enough. Which material is it?"},
    "none": {"de": "Keins davon", "en": "None of these"},
    "pick_any": {"de": "Material wählen", "en": "Choose the material"},
    "search_label": {"de": "Was willst du wegwerfen? (z. B. Pizzakarton, Teebeutel, Batterie)", "en": "What do you want to throw away? (e.g. pizza box, tea bag, battery)"},
    "no_hits": {"de": "Nichts gefunden. Probier ein anderes Wort oder das BSR-Abfall-ABC:", "en": "Nothing found. Try another word or the BSR A–Z:"},
    "matches": {"de": "Treffer", "en": "Matches"},
    "verified_true": {"de": "✅ Belegt durch offizielle Quelle", "en": "✅ Verified by an official source"},
    "verified_partial": {"de": "⚠️ Empfehlung – bitte im BSR-Abfall-ABC prüfen", "en": "⚠️ Guidance – please check the BSR A–Z"},
    "verified_false": {"de": "❓ Keine geprüfte Regel – bitte im BSR-Abfall-ABC nachsehen", "en": "❓ No verified rule – please check the BSR A–Z"},
    "source": {"de": "Quelle", "en": "Source"},
    "bsr": {"de": "BSR-Abfall-ABC", "en": "BSR A–Z"},
    "restart": {"de": "Neuer Gegenstand", "en": "New item"},
    "material": {"de": "Material", "en": "Material"},
    "footer": {"de": "Regeln Stand 2026-11-02 aus offiziellen Quellen (Berliner Senat, BSR-Liste, Umweltbundesamt, Verbraucherzentrale). Nicht verbunden mit BSR, ALBA oder dem Senat. Im Zweifel gilt das BSR-Abfall-ABC.",
               "en": "Rules as of 2026-11-02 from official sources (Berlin Senate, BSR list, German Environment Agency, consumer advice centre). Not affiliated with BSR, ALBA or the Senate. When in doubt, the BSR A–Z applies."},
    "model_note": {"de": "Wahrscheinlichkeiten sind kalibriert (Temperature Scaling). Unter {t:.0%} fragt die App nach.",
                   "en": "Probabilities are calibrated (temperature scaling). Below {t:.0%} the app asks you to confirm."},
}

BSR_AZ = "https://www.bsr.de/abfall-abc"

# ---------- state ----------
ss = st.session_state
ss.setdefault("lang", "en")
# Each tab keeps its own flow: {"material", "answers", "item"}.
ss.setdefault("photo", {"material": None, "answers": {}, "item": None})
ss.setdefault("text", {"material": None, "answers": {}, "item": None})
ss.setdefault("image_key", None)


def t(key: str, **kw) -> str:
    s = T[key][ss.lang]
    return s.format(**kw) if kw else s


def mname(m: str) -> str:
    return MATERIAL_NAMES[m][ss.lang]


def reset(ctx: str) -> None:
    ss[ctx] = {"material": None, "answers": {}, "item": None}


def choose_material(ctx: str, m: str) -> None:
    ss[ctx] = {"material": m, "answers": {}, "item": None}


@st.cache_resource(show_spinner="Loading model…")
def _head():
    return load_head()


@st.cache_resource(show_spinner="Loading image model (first run downloads ~600 MB)…")
def _warm_backbone(name: str):
    from wohin.embed import load_backbone

    return load_backbone(name)


@st.cache_data(show_spinner=False, max_entries=32)
def _predict(image_bytes: bytes):
    import io

    bundle = _head()
    _warm_backbone(bundle["backbone"])
    return predict(Image.open(io.BytesIO(image_bytes)), k=3, bundle=bundle)


# ---------- rendering ----------
def bin_card(o, hint: dict | None = None) -> None:
    lang = ss.lang
    badge = {"true": t("verified_true"), "partial": t("verified_partial"), "false": t("verified_false")}[o.verified]
    text = hint[lang] if hint else o.text[lang]
    src = f'<a href="{o.source_url}" target="_blank">{t("source")}</a>'
    bsr = "" if o.verified == "true" else f' · <a href="{o.bsr_az}" target="_blank"><b>{t("bsr")}</b></a>'
    if hint:  # unverified lexicon item: only BSR link plus the verified facts in the hint
        src = f'<a href="{o.bsr_az}" target="_blank"><b>{t("bsr")}</b></a>'
        bsr = ""
    border = o.colour if o.colour.upper() != "#EDEDED" else "#9E9E9E"
    st.markdown(
        f"""
<div style="border:3px solid {border};border-radius:14px;padding:16px 18px;margin:8px 0 4px 0;">
  <div style="display:flex;align-items:center;gap:14px;">
    <div aria-hidden="true" style="width:46px;height:46px;border-radius:10px;background:{o.colour};border:1px solid #888;flex:none;"></div>
    <div style="font-size:1.3rem;font-weight:700;line-height:1.3;">{html.escape(o.bin_label[lang])}</div>
  </div>
  <div style="margin-top:10px;">{html.escape(text)}</div>
  <div style="margin-top:10px;font-size:0.9rem;">{badge} · {src}{bsr}</div>
</div>""",
        unsafe_allow_html=True,
    )


def follow_up(ctx: str) -> None:
    """Ask questions until the rule engine reaches an outcome, then show the bin card."""
    state = ss[ctx]
    m, item = state["material"], state["item"]
    if item is not None and not item.verified:
        bin_card(get_outcome("check_bsr"), hint=item.hint)
        return
    q = next_question(m, state["answers"])
    if q is None:
        bin_card(outcome(m, state["answers"]))
        return
    st.markdown(f"**{q.text[ss.lang]}**")
    cols = st.columns(len(q.options))
    for col, (val, label) in zip(cols, q.options.items()):
        if col.button(label[ss.lang], key=f"{ctx}_q_{q.id}_{val}", use_container_width=True):
            state["answers"] = {**state["answers"], q.id: val}
            st.rerun()


def material_picker(options: list[str]) -> None:
    cols = st.columns(len(options))
    for col, m in zip(cols, options):
        if col.button(mname(m), key=f"pick_{m}", use_container_width=True,
                      type="primary" if ss.photo["material"] == m else "secondary"):
            choose_material("photo", m)
            st.rerun()


# ---------- layout ----------
head_cols = st.columns([4, 1])
head_cols[0].title("♻️ Wohin")
lang = head_cols[1].radio("Sprache / Language", ["en", "de"], index=["en", "de"].index(ss.lang),
                          format_func=lambda x: {"de": "DE", "en": "EN"}[x], horizontal=True, label_visibility="collapsed")
if lang != ss.lang:
    ss.lang = lang
    st.rerun()
st.caption(t("tagline"))

tab_photo, tab_text = st.tabs([t("tab_photo"), t("tab_text")])

with tab_photo:
    st.caption(t("one_item"))
    cam = st.camera_input(t("camera"), label_visibility="collapsed")
    up = st.file_uploader(t("upload"), type=["jpg", "jpeg", "png", "webp"])
    file = cam or up
    if file is not None:
        data = file.getvalue()
        key = hashlib.sha1(data).hexdigest()
        if ss.image_key != key:
            reset("photo")
            ss.image_key = key
        bundle = _head()
        top = _predict(data)
        c1, c2 = st.columns([1, 1.4])
        c1.image(data, use_container_width=True)
        with c2:
            st.markdown(t("top3"))
            for m, p in top:
                st.progress(min(max(p, 0.0), 1.0), text=f"{mname(m)} — {p:.0%}")
            st.caption(t("model_note", t=bundle["threshold"]))
        confident = is_confident(top, bundle)
        if confident and ss.photo["material"] is None:
            choose_material("photo", top[0][0])
        if confident:
            st.markdown(t("confident", m=mname(top[0][0])))
        else:
            st.warning(t("unsure"))
        material_picker([m for m, _ in top])
        if st.checkbox(t("none"), key="none_of_these"):
            st.selectbox(t("pick_any"), list(MATERIAL_NAMES), format_func=mname, key="any_material", index=None,
                         on_change=lambda: choose_material("photo", ss.any_material) if ss.any_material else None)
        if ss.photo["material"]:
            st.divider()
            st.markdown(f"{t('material')}: **{mname(ss.photo['material'])}**")
            follow_up("photo")
            st.button(t("restart"), on_click=reset, args=("photo",), key="restart_photo")

with tab_text:
    query = st.text_input(t("search_label"), key="query")
    if query:
        hits = search(query)
        if not hits:
            st.info(t("no_hits"))
            st.markdown(f"[{t('bsr')}]({BSR_AZ})")
        else:
            labels = {h[0].id: f"{h[0].name(ss.lang)} · {mname(h[1])}" for h in hits}
            chosen = st.radio(t("matches"), list(labels), format_func=labels.get, key=f"hit_{query}")
            item = next(h[0] for h in hits if h[0].id == chosen)
            if ss.text["item"] is None or ss.text["item"].id != item.id:
                ss.text = {"material": item.material, "answers": dict(item.answers), "item": item}
            st.markdown(f"{t('material')}: **{mname(item.material)}**")
            follow_up("text")

st.divider()
st.caption(t("footer"))
st.caption("Wohin · Neural Storm 2026 · GP-05 · [GitHub](https://github.com/hoongyu/wohin)")
