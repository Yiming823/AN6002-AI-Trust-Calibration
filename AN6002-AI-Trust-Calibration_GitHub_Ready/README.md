# Calibrating Trust in AI — Interactive Demo

**AN6002 Group Project · Section C · Group 5**

This repository hosts the **AI Trust Calibration Checker**, an illustrative Streamlit prototype developed for the *Calibrating Trust in AI* group project.

## What the demo does

Given seven inputs describing an AI response and its user, a pre-trained logistic-regression pipeline estimates **over-trust risk** in the study dataset. In this project, over-trust means that the user's trust score (0–10, multiplied by 10) exceeds the answer's accuracy (0–100%) by **more than 20 points**.

Inputs: AI-stated confidence, cited sources, hedging words, question topic, decision importance, AI familiarity, and digital literacy.

The app displays the estimated risk, compares it with the **24.5%** dataset average, and displays a screening message using a **30%** exploratory threshold.

**Important:** This is an educational demonstration based on illustrative data whose collection provenance is not independently verified. The risk estimate is **not** the probability an answer is wrong; it has not been validated for real-world decisions. The reported logistic-regression cross-validated AUC in the project is approximately **0.768**. Associations are not causal effects.

## Files

```text
.
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── models/
    ├── overtrust_model.joblib
    └── overtrust_meta.json
```

The model artifacts were saved by the group's analysis notebook. **Running the notebook is not required to start this app**; the trained model is included in `models/`. The full research notebook, source datasets, and figures are not included in this demo-only deployment repository.

## Run locally

Python **3.12** is recommended, matching the suggested Streamlit Community Cloud deployment setting.

```bash
# Open a terminal in the repository root.
python3.12 -m venv .venv
source .venv/bin/activate  # macOS/Linux
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

**Model compatibility:** The saved `overtrust_meta.json` records `scikit-learn 1.9.1`. This repository pins that version to reduce model serialization incompatibilities. If changing it, regenerate the model with the intended version and retest before deployment.

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub (Public if you want the deployment to be openly discoverable).
2. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select this repository, branch `main`, and **Main file path:** `app.py`.
4. In **Advanced settings**, select **Python 3.12**.
5. Deploy, then open the public `.streamlit.app` URL to verify predictions.

A Google Drive link to `app.py` opens the **source file**, not an interactive website. Use the **deployed** Streamlit URL (not a Drive file URL) for the e-poster QR code.

## Credits

Section C Group 5 — Zhang Yueyang, Zeng Min, Chen Boyu, Zhao Shuman, Guo Yiming.
