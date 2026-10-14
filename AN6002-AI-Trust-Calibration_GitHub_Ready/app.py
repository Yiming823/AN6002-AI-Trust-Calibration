"""
Over-trust risk checker: a minimal demo for the AN6002 group project.

It loads the logistic regression saved by the analysis notebook and shows the
estimated chance that an AI response will be OVER-TRUSTED, i.e. that a user's
trust (0-10, x10) exceeds the response's real accuracy by more than 20 points.

Run:  streamlit run app.py
"""
import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Trust Calibration Checker", page_icon=None)

MODEL_DIR = Path(__file__).resolve().parent / "models"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "overtrust_model.joblib")
    with (MODEL_DIR / "overtrust_meta.json").open(encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


try:
    model, meta = load_model()
except FileNotFoundError:
    st.error("Required model files were not found. Make sure the models/ folder "
             "contains overtrust_model.joblib and overtrust_meta.json in the deployed repository.")
    st.stop()

opts = meta["options"]
lo, hi = meta["confidence_range"]
base = meta["base_rate"]
screening_threshold = meta.get("screening_threshold", 0.30)

st.title("AI Trust Calibration Checker")
st.caption(
    "Early-warning demo for identifying AI responses that may be trusted "
    "beyond what their accuracy warrants. Built on simulated data; not for real decisions."
)
st.markdown(
    "**What this shows:** for an AI response with the characteristics you choose, the estimated chance that "
    f"users would trust it **much more than its accuracy deserves** (trust x 10 exceeds accuracy by more than "
    f"{meta['threshold_points']} points). It does **not** say whether the answer is right or wrong."
)

st.subheader("Describe the AI response")
c1, c2 = st.columns(2)
with c1:
    topic = st.selectbox("Question topic", opts["query_category"],
                         index=opts["query_category"].index("general_knowledge"),
                         format_func=lambda x: x.replace("_", " ").capitalize())
    confidence = st.slider("Confidence the AI states (%)", int(lo), int(hi), 75)
    cites = st.checkbox("Response cites sources", value=False)
    hedges = st.checkbox("Response uses hedging words (e.g. 'might', 'possibly')", value=False)
with c2:
    importance = st.selectbox("Importance of the decision", opts["decision_importance"],
                              index=opts["decision_importance"].index("Medium"))
    familiarity = st.selectbox("User's familiarity with AI", opts["ai_familiarity_level"],
                               index=opts["ai_familiarity_level"].index("Intermediate"))
    literacy = st.selectbox("User's digital literacy", opts["digital_literacy_score"],
                            index=opts["digital_literacy_score"].index("Medium"))

row = pd.DataFrame([{
    "ai_confidence_percentage": float(confidence),
    "has_cited_sources": int(cites),
    "contains_hedging_words": int(hedges),
    "query_category": topic,
    "decision_importance": importance,
    "ai_familiarity_level": familiarity,
    "digital_literacy_score": literacy,
}])[meta["features"]]

prob = float(model.predict_proba(row)[0, 1])

st.subheader("Result")

shown = (
    "under 1%" if prob < 0.01
    else "over 99%" if prob > 0.99
    else f"{prob:.0%}"
)

st.metric(
    "Estimated chance of over-trust",
    shown,
    delta=f"{(prob - base) * 100:+.0f} points vs dataset average ({base:.0%})",
    delta_color="inverse"
)

if prob >= screening_threshold:
    st.warning(
        f"Higher screening risk. The predicted probability exceeds the "
        f"{screening_threshold:.0%} practical screening threshold."
    )

    st.markdown(
        "**Recommended calibration action:** "
        "Add stronger verification cues, such as a source-checking prompt, "
        "reliability indicator, uncertainty information, or human review for high-stakes decisions."
    )

elif prob >= base:
    st.info(
        "Moderate over-trust risk. The predicted probability is above the dataset average "
        "but below the practical screening threshold."
    )

    st.markdown(
        "**Recommended calibration action:** "
        "Consider showing uncertainty or reminding the user to verify important claims."
    )

else:
    st.success(
        "Lower over-trust risk. The predicted probability is below the dataset average."
    )

    st.markdown(
        "**Recommended calibration action:** "
        "Standard transparency and source-verification cues are likely sufficient."
    )


with st.expander("How to read this, and its limits"):
    st.markdown(
        f"""
- **What the number is:** the share of similar responses in our dataset that were over-trusted, according to a
  logistic regression fitted to {meta['n_rows']:,} simulated user-AI interactions. The dataset average is {base:.0%}.
- **What it is not:** it is not the chance the answer is wrong, and not a measure of how much any real person trusts any real AI.
- **Over-trust is approximate:** it compares trust (0-10, multiplied by 10) with accuracy (0-100%), which assumes one trust point
  equals ten accuracy points. The 20-point cut-off is a judgement call.
- **Very high or very low values:** appear when several risk factors combine. Read the result as a relative ranking, not a precise probability.
- **Accuracy of the model:** it ranks risk only moderately well (cross-validated AUC about {meta['cv_auc_logistic']:.2f}, where 0.5 is guessing).
- **Screening threshold:** a predicted risk of {screening_threshold:.0%} is used as a practical early-warning threshold. Among the thresholds examined in the hold-out sample, 30% produced the highest observed F1-score and improved recall. It should not be interpreted as a universal or optimal cut-off.
- **Patterns, not causes:** the inputs are associated with over-trust in this data; changing them may not change real trust.
- **Synthetic data:** all results come from simulated datasets.
"""
    )
