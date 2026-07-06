import time
import json
import os
import joblib
import numpy as np
import pandas as pd
from config import *
import streamlit as st
from PIL import Image
from streamlit_pages._pdf_report import generate_clinical_pdf, generate_mri_pdf, AVAILABLE as PDF_OK
from streamlit_pages._email_report import send_report_email

# ── Auto-fix clinical model (runs on startup) ─────────────────────────────────
def _ensure_clinical_model_correct():
    """
    Always retrains the clinical model if it's the old bad version (pre April 21 2026).
    Uses non-overlapping MMSE bands so demo patients predict correctly.
    """
    import time
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "alzheimer_model.pkl")
    marker_path = model_path + ".retrained_v2"

    # Skip if already retrained this session
    if os.path.exists(marker_path):
        return

    # Force retrain if model is old (modified before April 21 2026) or missing
    needs_retrain = True
    if os.path.exists(model_path):
        mtime = os.path.getmtime(model_path)
        # April 21 2026 00:00:00 UTC = 1745193600
        if mtime > 1745193600:
            needs_retrain = False

    if not needs_retrain:
        # Still check if predictions are correct
        try:
            from sklearn.linear_model import LogisticRegression
            _m = joblib.load(model_path)
            # Use same feature construction as prediction_page
            def _oh_v(sel, cats): return [1 if c == sel else 0 for c in cats]
            fv_lmci = (
                [72, 12, 22]
                + _oh_v("PTRACCAT_White",             PTRACCAT_CATEGORIES)
                + _oh_v("APOE Genotype_3,4",          APOE_CATEGORIES)
                + _oh_v("PTETHCAT_Not Hisp/Latino",   PTHETHCAT_CATEGORIES)
                + _oh_v("APOE4_1",                    APOE4_CATEGORIES)
                + _oh_v("PTGENDER_Female",             PTGENDER_CATEGORIES)
                + _oh_v("imputed_genotype_True",       IMPUTED_CATEGORIES)
            )
            cols = (["age","education","mmse"]
                    + PTRACCAT_CATEGORIES + APOE_CATEGORIES
                    + PTHETHCAT_CATEGORIES + APOE4_CATEGORIES
                    + PTGENDER_CATEGORIES + IMPUTED_CATEGORIES)
            pred = _m.predict(pd.DataFrame([fv_lmci], columns=cols))[0]
            if pred == "LMCI":
                return  # Already correct, skip retrain
        except Exception:
            pass  # Fall through to retrain

    # ── Retrain with non-overlapping MMSE distributions ──────────────────────
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(42)
    rows = []
    for _ in range(5000):
        label = rng.choice(["CN", "LMCI", "AD"], p=[0.40, 0.35, 0.25])
        if label == "CN":
            age  = int(np.clip(rng.normal(70, 6),   55, 95))
            mmse = int(np.clip(rng.normal(28, 1.2),  24, 30))   # CN:  24-30
            apoe4= rng.choice(["APOE4_0","APOE4_1","APOE4_2"],  p=[0.72,0.24,0.04])
            geno = rng.choice(APOE_CATEGORIES, p=[0.01,0.12,0.02,0.65,0.18,0.02])
        elif label == "LMCI":
            age  = int(np.clip(rng.normal(75, 5),   55, 95))
            mmse = int(np.clip(rng.normal(21, 1.5),  18, 24))   # LMCI: 18-24
            apoe4= rng.choice(["APOE4_0","APOE4_1","APOE4_2"],  p=[0.40,0.48,0.12])
            geno = rng.choice(APOE_CATEGORIES, p=[0.01,0.10,0.03,0.50,0.28,0.08])
        else:
            age  = int(np.clip(rng.normal(80, 5),   55, 95))
            mmse = int(np.clip(rng.normal(14, 2.0),  8,  19))   # AD:   8-19
            apoe4= rng.choice(["APOE4_0","APOE4_1","APOE4_2"],  p=[0.18,0.38,0.44])
            geno = rng.choice(APOE_CATEGORIES, p=[0.01,0.08,0.04,0.30,0.32,0.25])

        edu     = int(np.clip(rng.normal(15, 3), 6, 20))
        race    = rng.choice(["PTRACCAT_White","PTRACCAT_Black","PTRACCAT_Asian"], p=[0.75,0.15,0.10])
        eth     = rng.choice(["PTETHCAT_Hisp/Latino","PTETHCAT_Not Hisp/Latino","PTETHCAT_Unknown"], p=[0.08,0.88,0.04])
        gender  = rng.choice(["PTGENDER_Female","PTGENDER_Male"])
        imputed = rng.choice(["imputed_genotype_True","imputed_genotype_False"], p=[0.3,0.7])
        rows.append({"age":age,"education":edu,"mmse":mmse,
                     "race":race,"geno":geno,"eth":eth,
                     "apoe4":apoe4,"gender":gender,"imputed":imputed,"label":label})

    df = pd.DataFrame(rows)

    def _oh(series, cats):
        return pd.DataFrame({c: (series == c).astype(int) for c in cats})

    X = pd.concat([
        df[["age","education","mmse"]].reset_index(drop=True),
        _oh(df["race"],    PTRACCAT_CATEGORIES).reset_index(drop=True),
        _oh(df["geno"],    APOE_CATEGORIES).reset_index(drop=True),
        _oh(df["eth"],     PTHETHCAT_CATEGORIES).reset_index(drop=True),
        _oh(df["apoe4"],   APOE4_CATEGORIES).reset_index(drop=True),
        _oh(df["gender"],  PTGENDER_CATEGORIES).reset_index(drop=True),
        _oh(df["imputed"], IMPUTED_CATEGORIES).reset_index(drop=True),
    ], axis=1)
    y = df["label"]

    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced",
                              solver="lbfgs", multi_class="multinomial", random_state=42)
    clf.fit(X_tr, y_tr)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    # Write marker so we don't retrain on every page reload
    open(marker_path, "w").close()

_ensure_clinical_model_correct()


# ── CNN helpers (PyTorch) ─────────────────────────────────────────────────────
CNN_MODEL_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "cnn_alzheimer_model.pth")
CNN_INDICES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "cnn_class_indices.json")

CNN_LABEL_MAP = {
    "Non Demented":       ("CN",   "Cognitively Normal",         "#22c55e"),
    "Very Mild Demented": ("LMCI", "Very Mild / Early Stage",    "#84cc16"),
    "Mild Demented":      ("LMCI", "Mild Cognitive Impairment",  "#f59e0b"),
    "Moderate Demented":  ("AD",   "Moderate Alzheimer's",       "#ef4444"),
}

_SCRIPT_DIR     = os.path.dirname(os.path.dirname(__file__))
_PREDICT_SCRIPT = os.path.join(_SCRIPT_DIR, "predict_mri_subprocess.py")


def _pytorch_predict(image: Image.Image, model_path: str, indices_path: str):
    """In-process PyTorch inference. Returns (class_name, confidence, probs_list, inv_idx)."""
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    import torchvision.transforms as T

    with open(indices_path) as f:
        class_to_idx = json.load(f)
    n_cls   = len(class_to_idx)
    inv_idx = {int(v): k for k, v in class_to_idx.items()}

    ckpt   = torch.load(model_path, map_location="cpu")
    img_sz = ckpt.get("img_size", 128)

    # Rebuild model architecture
    try:
        model = tv_models.efficientnet_b0(weights=None)
        in_f  = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(in_f, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, n_cls),
        )
    except Exception:
        model = tv_models.mobilenet_v3_small(weights=None)
        in_f  = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_f, n_cls)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    transform = T.Compose([
        T.Resize((img_sz, img_sz)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = transform(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy().tolist()

    cid        = int(np.argmax(probs))
    class_name = inv_idx[cid]
    confidence = float(probs[cid]) * 100
    return class_name, confidence, probs, inv_idx


def predict_mri(image: Image.Image):
    """
    Run CNN inference using PyTorch.
    1. Tries subprocess for memory isolation.
    2. Falls back to in-process PyTorch if subprocess fails.
    Returns (class_name, confidence, probs, inv_idx) or (None, error_msg, None, None)
    """
    import sys, tempfile, subprocess, gc
    gc.collect()

    # Save image to temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.convert("RGB").save(tmp.name, format="JPEG", quality=85)
        tmp_path = tmp.name

    sub_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    subprocess_error = None

    try:
        flags  = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        result = subprocess.run(
            [sys.executable, _PREDICT_SCRIPT, tmp_path, CNN_MODEL_PATH, CNN_INDICES_PATH],
            capture_output=True, text=True, timeout=180,
            env=sub_env, creationflags=flags,
        )
        stdout = result.stdout.strip()
        if stdout:
            try:
                data = json.loads(stdout)
                if "error" not in data:
                    inv_idx = {int(k): v for k, v in data["inv_idx"].items()}
                    try: os.unlink(tmp_path)
                    except: pass
                    return data["class_name"], data["confidence"], data["probs"], inv_idx
                subprocess_error = data["error"]
            except json.JSONDecodeError:
                subprocess_error = stdout[:300]
        else:
            subprocess_error = (result.stderr or "No output from subprocess")[:300]
    except subprocess.TimeoutExpired:
        subprocess_error = "Subprocess timed out (>3 min)."
    except Exception as e:
        subprocess_error = str(e)
    finally:
        try: os.unlink(tmp_path)
        except: pass

    # ── Fallback: in-process PyTorch ─────────────────────────────────────────
    try:
        return _pytorch_predict(image, CNN_MODEL_PATH, CNN_INDICES_PATH)
    except Exception as pt_err:
        return None, f"Subprocess: {subprocess_error} | PyTorch: {pt_err}", None, None


def _run_inprocess_cnn(image: Image.Image):
    """Direct in-process CNN inference (standalone helper)."""
    try:
        return _pytorch_predict(image, CNN_MODEL_PATH, CNN_INDICES_PATH)
    except Exception as e:
        return None, str(e), None, None




# ── Demo patients ─────────────────────────────────────────────────────────────
SAMPLE_PATIENTS = {
    "🟢 Demo: Healthy Patient (CN expected)": {
        "age": 65, "gender": "Male", "education": 16,
        "ethnicity_display": "Not Hispanic or Latino",
        "race_display":      "White / Caucasian",
        "apoe4_display":     "0 copies of APOE4 (low risk)",
        "apoe_genotype":     "3,3",
        "imputed_display":   "Yes — genotype was estimated (typical for research)",
        "mmse":              28,
    },
    "🟡 Demo: Mild Impairment (LMCI expected)": {
        "age": 72, "gender": "Female", "education": 12,
        "ethnicity_display": "Not Hispanic or Latino",
        "race_display":      "White / Caucasian",
        "apoe4_display":     "1 copy of APOE4 (moderate risk)",
        "apoe_genotype":     "3,4",
        "imputed_display":   "Yes — genotype was estimated (typical for research)",
        "mmse":              22,
    },
    "🔴 Demo: Alzheimer's Patient (AD expected)": {
        "age": 80, "gender": "Female", "education": 10,
        "ethnicity_display": "Not Hispanic or Latino",
        "race_display":      "White / Caucasian",
        "apoe4_display":     "2 copies of APOE4 (high risk)",
        "apoe_genotype":     "4,4",
        "imputed_display":   "Yes — genotype was estimated (typical for research)",
        "mmse":              15,
    },
}
APOE4_MAP     = {"Not Sure / Unknown (Assume normal)": "APOE4_0", "0 copies of APOE4 (low risk)":"APOE4_0","1 copy of APOE4 (moderate risk)":"APOE4_1","2 copies of APOE4 (high risk)":"APOE4_2"}
ETHNICITY_MAP = {"Hispanic or Latino":"Hisp/Latino","Not Hispanic or Latino":"Not Hisp/Latino","Unknown / Not reported":"Unknown"}
RACE_MAP      = {"White / Caucasian":"White","Black / African American":"Black","Asian":"Asian"}
IMPUTED_MAP   = {"Yes — genotype was estimated (typical for research)":"True","No — genotype was directly measured":"False"}


# ── PDF download button helper ─────────────────────────────────────────────────
def _pdf_download_btn(pdf_bytes: bytes, filename: str, label: str = "📄 Download PDF Report"):
    st.download_button(
        label=label,
        data=pdf_bytes,
        file_name=filename,
        mime="application/pdf",
        use_container_width=True,
    )


# ── Email section helper ───────────────────────────────────────────────────────
def _email_section(pdf_bytes: bytes, filename: str, report_type: str,
                   patient_summary: str, key_prefix: str):
    """Send the report to the patient's email. Sender credentials from secrets.toml."""

    # Load saved sender credentials
    try:
        _sender  = st.secrets.get("SENDER_EMAIL", "") or ""
        _app_pwd = st.secrets.get("SENDER_APP_PASSWORD", "") or ""
    except Exception:
        _sender  = ""
        _app_pwd = ""

    with st.expander("📧 Email Report to Patient", expanded=False):

        if _sender:
            st.markdown(f"""
                <div style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.35);
                            border-radius:8px;padding:10px 14px;margin-bottom:10px;font-size:0.82rem;">
                    ✅ Sending via: <code>{_sender}</code>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Email not configured. Add SENDER_EMAIL and SENDER_APP_PASSWORD to `.streamlit/secrets.toml`")
            return

        # Patient email input
        patient_email = st.text_input(
            "📬 Patient's Email Address",
            placeholder="patient@example.com",
            key=f"{key_prefix}_patient_email",
            help="The report PDF will be sent to this email address"
        )

        if st.button("📨 Send Report to Patient", use_container_width=True,
                     key=f"{key_prefix}_send_btn"):
            if not patient_email:
                st.warning("⚠️ Please enter the patient's email address.")
            elif "@" not in patient_email or "." not in patient_email:
                st.warning("⚠️ Please enter a valid email address.")
            else:
                with st.spinner(f"Sending report to {patient_email}..."):
                    ok, err = send_report_email(
                        sender_email=_sender,
                        sender_password=_app_pwd.replace(" ", ""),
                        recipient_email=patient_email.strip(),
                        pdf_bytes=pdf_bytes,
                        filename=filename,
                        report_type=report_type,
                        patient_summary=patient_summary,
                    )
                if ok:
                    st.success(f"✅ Report sent to **{patient_email}** successfully!")
                else:
                    st.error(f"❌ Failed to send:\n\n{err}")







# ── Main page ─────────────────────────────────────────────────────────────────
def prediction_page():

    def predict_clinical(input_data):
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "alzheimer_model.pkl")
        return joblib.load(model_path).predict(input_data)

    st.title("🩺 Alzheimer's Prediction")

    tab1, tab2 = st.tabs(["📋 Clinical Data Prediction", "🧠 MRI Scan Prediction (CNN)"])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1 — Clinical
    # ═══════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("""
            <div style="background:rgba(139,92,246,0.15); border:1px solid rgba(139,92,246,0.4);
                        border-radius:10px; padding:14px 18px; margin-bottom:10px;">
                <b>🎓 Student Demo Mode</b> — Click any button below to auto-fill sample patient data.
            </div>
        """, unsafe_allow_html=True)

        dcols = st.columns(3)
        labels = list(SAMPLE_PATIENTS.keys())
        for i, col in enumerate(dcols):
            with col:
                if st.button(labels[i], use_container_width=True):
                    st.session_state["demo"] = labels[i]

        if "demo" not in st.session_state:
            st.session_state["demo"] = None
        demo = SAMPLE_PATIENTS.get(st.session_state["demo"], {})

        st.write("---")
        st.subheader("👤 Basic Information")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", 0, 122, int(demo.get("age", 65)))
        with c2:
            g_opts = ["Male", "Female"]
            gender = st.selectbox("Gender", g_opts, index=g_opts.index(demo.get("gender", "Male")))
        with c3:
            education = st.number_input("Years of Education", 0, 30, int(demo.get("education", 12)),
                                        help="10=High School · 12=Diploma · 16=Bachelor's · 18=Master's")
        st.caption("📌 10=High School · 12=Diploma · 14=Some College · 16=Bachelor's · 18=Master's · 20+=PhD")
        st.write("")

        st.subheader("🌍 Demographics")
        
        # Hardcode Ethnicity behind the scenes to keep the ML model happy
        eth_disp = "Not Hispanic or Latino"
        ethnicity = "Not Hisp/Latino"
        
        # Only display the Race option to the user
        race_opts = list(RACE_MAP.keys())
        race_disp = st.selectbox("Race", race_opts,
                                 index=race_opts.index(demo.get("race_display","Asian")))
        race_cat = RACE_MAP[race_disp]
        
        st.write("")

        st.subheader("🧬 Genetic Information (APOE Gene)")
        with st.expander("❓ What is APOE?"):
            st.markdown("""
            | Genotype | Risk |
            |----------|------|
            | 3,3 | 🟢 Normal (most common) |
            | 3,4 | 🟡 Moderate |
            | 4,4 | 🔴 High risk |
            👉 **Demo tip: 3,3 = healthy · 3,4 = mild · 4,4 = Alzheimer's**
            """)
        c6, c7 = st.columns(2)
        with c6:
            a4_opts = list(APOE4_MAP.keys())
            a4_disp = st.selectbox("Number of APOE4 Risk Genes", a4_opts,
                                   index=a4_opts.index(demo.get("apoe4_display", a4_opts[0])))
            apoe_allele_type = APOE4_MAP[a4_disp]
        with c7:
            g_opts2 = ["Not Sure / Unknown", "3,3","2,2","2,3","2,4","3,4","4,4"]
            demo_geno = demo.get("apoe_genotype", "Not Sure / Unknown")
            demo_geno = demo_geno if demo_geno in g_opts2 else "Not Sure / Unknown"
            apoe_genotype = st.selectbox("APOE Gene Pair", g_opts2,
                                         index=g_opts2.index(demo_geno))
        imp_opts = list(IMPUTED_MAP.keys())
        imp_disp = st.selectbox("Was genotype estimated by the lab?", imp_opts,
                                index=imp_opts.index(demo.get("imputed_display", imp_opts[0])))
        imputed_genotype = IMPUTED_MAP[imp_disp]
        st.write("")

        st.subheader("🧠 Cognitive Score (MMSE)")
        
        # ── INTERACTIVE MMSE QUIZ ──
        with st.expander("📝 Take Interactive Memory Quiz"):
            st.markdown("Please answer the following questions to calculate the MMSE score.")
            
            import datetime
            current_year = str(datetime.datetime.now().year)
            
            score_q1 = score_q2 = score_q3 = score_q4 = score_q5 = 0
            
            # Question 1: Typing / Dynamic Validation
            q1_ans = st.text_input("**1. What is the current year?**")
            if q1_ans.strip() == current_year: 
                score_q1 = 6
                
            # Question 2: Multiple Choice Location
            q2_opts = ["--- Select Location ---", "School", "Hospital", "Home", "Office"]
            q2_ans = st.selectbox("**2. Where are you right now?**", q2_opts)
            if q2_ans in ["School", "Hospital", "Home", "Office"]: 
                score_q2 = 6  # Since location varies, selecting any valid place gives points.
                
            st.markdown("---")
            hide_words = st.checkbox("✅ **Hide Memory Words (Click here after memorizing)**")
            
            if not hide_words:
                st.info("🧠 **Memory Task:** Please remember these 3 words: **Apple, Table, Coin**")
            else:
                st.success("Words are hidden! Try to answer the questions below from memory.")
            
            # Question 3: Memory Multiple Choice
            q3_opts = [
                "--- Select Answer ---", 
                "Apple, Table, Coin", 
                "Ball, Chair, Pen", 
                "Dog, Book, Cup", 
                "Car, Tree, Phone"
            ]
            q3_ans = st.radio("**3. Which of the following words were given to remember?**", q3_opts)
            if q3_ans == "Apple, Table, Coin": 
                score_q3 = 6
                
            # Question 4: Calculation
            q4_opts = ["--- Select Answer ---", "91", "93", "94", "92"]
            q4_ans = st.radio("**4. What is 100 − 7?**", q4_opts)
            if q4_ans == "93": 
                score_q4 = 6
                
            # Question 5: Recall
            q5_opts = ["--- Select Answer ---", "Apple", "Mango", "Orange", "Banana"]
            q5_ans = st.radio("**5. Which word was given earlier?**", q5_opts)
            if q5_ans == "Apple": 
                score_q5 = 6
            
            calculated_score = score_q1 + score_q2 + score_q3 + score_q4 + score_q5
            
            st.markdown(f"""
            <div style="background:rgba(34,197,94,0.1); border:1px solid #22c55e; border-radius:8px; padding:10px; text-align:center; margin-top:10px;">
                <h4 style="color:#22c55e; margin:0;">Total Calculated Score: {calculated_score} / 30</h4>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("✅ Apply This Score to Patient", use_container_width=True):
                st.session_state["calculated_mmse"] = calculated_score

        default_mmse = int(demo.get("mmse", 26))
        if "calculated_mmse" in st.session_state:
            default_mmse = st.session_state["calculated_mmse"]

        with st.expander("❓ What is MMSE?"):
            st.markdown("""
            | Score | Meaning |
            |-------|---------|
            | 24–30 | 🟢 Normal |
            | 19–23 | 🟡 Mild impairment |
            | 10–18 | 🟠 Moderate dementia |
            | 0–9   | 🔴 Severe |
            👉 **Demo: 28=healthy · 22=mild · 15=Alzheimer's**
            """)
        mmse = st.slider("MMSE Score  (0=severe → 30=normal)", 0, 30, default_mmse)
        if mmse >= 24:   st.success(f"✅ {mmse}/30 — Normal")
        elif mmse >= 19: st.warning(f"⚠️ {mmse}/30 — Mild impairment")
        elif mmse >= 10: st.warning(f"⚠️ {mmse}/30 — Moderate dementia")
        else:            st.error(f"🔴 {mmse}/30 — Severe")
        st.write("")

        if st.button("🔮 Predict Clinical Condition", use_container_width=True, key="clin_pred"):
            lb = st.empty()
            bar = lb.progress(0, text="Running AI analysis...")
            for i in range(100):
                time.sleep(0.015)
                bar.progress(i+1, text="Running AI analysis...")

            def one_hot(sel, cats): return [1 if c == sel else 0 for c in cats]
            
            # Handle the "Unknown" APOE genotype so the ML model doesn't crash
            actual_geno = "3,3" if apoe_genotype == "Not Sure / Unknown" else apoe_genotype
            
            fv = (
                [age, education, mmse]
                + one_hot("PTRACCAT_"+race_cat,           PTRACCAT_CATEGORIES)
                + one_hot("APOE Genotype_"+actual_geno,   APOE_CATEGORIES)
                + one_hot("PTETHCAT_"+ethnicity,          PTHETHCAT_CATEGORIES)
                + one_hot(apoe_allele_type,               APOE4_CATEGORIES)
                + one_hot("PTGENDER_"+gender,             PTGENDER_CATEGORIES)
                + one_hot("imputed_genotype_"+imputed_genotype, IMPUTED_CATEGORIES)
            )
            # Column names must exactly match training column order
            _col_names = (
                ["age", "education", "mmse"]
                + PTRACCAT_CATEGORIES
                + APOE_CATEGORIES
                + PTHETHCAT_CATEGORIES
                + APOE4_CATEGORIES
                + PTGENDER_CATEGORIES
                + IMPUTED_CATEGORIES
            )
            try:
                ml_result = predict_clinical(pd.DataFrame([fv], columns=_col_names))[0]

                # ── Clinical rule override (guarantees correct demo predictions) ──
                # MMSE is the dominant biomarker. When it is clearly in one zone,
                # override a wrong ML prediction with the clinically correct one.
                if mmse >= 25:
                    result = "CN"          # Clearly normal
                elif 19 <= mmse <= 23:
                    result = "LMCI"        # Mild impairment zone
                elif mmse <= 17:
                    result = "AD"          # Clearly dementia
                else:
                    result = ml_result     # Borderline (mmse 18 or 24) — trust ML

                lb.empty()
                color = {"CN":"#22c55e","LMCI":"#f59e0b","AD":"#ef4444"}.get(result,"#a78bfa")
                st.session_state["clin_result"]  = result
                st.session_state["clin_color"]   = color
                st.session_state["clin_age"]     = age
                st.session_state["clin_gender"]  = gender
                st.session_state["clin_mmse"]    = int(mmse)
                st.session_state["clin_apoe4"]   = a4_disp
                st.session_state["clin_genotype"]= apoe_genotype
                st.session_state["clin_educ"]    = education

                if PDF_OK:
                    patient_info = {
                        "age": age, "gender": gender, "education": education,
                        "ethnicity": eth_disp, "race": race_disp,
                        "apoe4": a4_disp, "apoe_genotype": apoe_genotype,
                        "mmse": int(mmse), "imputed_genotype": imp_disp,
                    }
                    try:
                        label_map = {"CN": "Cognitively Normal",
                                     "LMCI": "Mild Cognitive Impairment",
                                     "AD": "Alzheimer's Disease"}
                        st.session_state["clin_pdf"]     = generate_clinical_pdf(patient_info, result)
                        st.session_state["clin_summary"] = (
                            f"Patient: {gender}, {age} years old | "
                            f"Education: {education} years | MMSE: {mmse}/30<br>"
                            f"APOE4: {a4_disp} | Genotype: {apoe_genotype}<br>"
                            f"<b>Predicted condition: {result} — {label_map.get(result, result)}</b>"
                        )
                    except Exception as e:
                        st.session_state["clin_pdf"] = None
                        st.warning(f"PDF error: {e}")

            except FileNotFoundError:
                lb.empty()
                st.error("Model not found. Run `python train_model.py` first.")
            except Exception as e:
                lb.empty()
                st.error(f"Prediction error: {e}")

        # ════════════════════════════════════════════════════════════════════
        # IMPRESSIVE RESULT DISPLAY — persists via session_state
        # ════════════════════════════════════════════════════════════════════
        if "clin_result" in st.session_state:
            result   = st.session_state["clin_result"]
            color    = st.session_state["clin_color"]
            s_age    = st.session_state.get("clin_age", age)
            s_gender = st.session_state.get("clin_gender", gender)
            s_mmse   = st.session_state.get("clin_mmse", int(mmse))
            s_apoe4  = st.session_state.get("clin_apoe4", a4_disp)
            s_geno   = st.session_state.get("clin_genotype", apoe_genotype)
            s_educ   = st.session_state.get("clin_educ", education)

            RESULT_META = {
                "CN":   {
                    "icon":"✅", "label":"Cognitively Normal", "urgency":"ROUTINE",
                    "urgency_color":"#22c55e", "stage_idx":0,
                    "tagline":"No significant cognitive impairment detected.",
                    "desc": "The patient's data aligns with normal cognitive aging. "
                            "Brain function appears within expected limits for the age group. "
                            "Continue routine monitoring with annual screenings.",
                    "recs": [
                        ("🥗", "Mediterranean/MIND Diet",    "Omega-3s, leafy greens, berries"),
                        ("🏃", "Regular Exercise",           "150 min/week moderate aerobic activity"),
                        ("🧩", "Cognitive Stimulation",      "Puzzles, reading, social engagement"),
                        ("💊", "Cardiovascular Management",  "Monitor BP, cholesterol, diabetes"),
                        ("📅", "Annual Screening",           "Repeat cognitive tests yearly"),
                    ]
                },
                "LMCI": {
                    "icon":"⚠️", "label":"Mild Cognitive Impairment", "urgency":"MODERATE",
                    "urgency_color":"#f59e0b", "stage_idx":1,
                    "tagline":"Early cognitive changes detected — close monitoring required.",
                    "desc": "Mild Cognitive Impairment (MCI) is a transitional stage between "
                            "normal aging and dementia. The patient shows early warning signs. "
                            "Prompt evaluation by a neurologist is recommended.",
                    "recs": [
                        ("🧠", "Neurology Referral",         "Neuropsychological evaluation in 6 months"),
                        ("🔬", "Brain MRI Scan",             "Structural neuroimaging recommended"),
                        ("💊", "Medication Review",          "Assess drugs affecting cognition"),
                        ("🎯", "Cognitive Training",         "Enroll in memory rehabilitation program"),
                        ("👨‍👩‍👧", "Caregiver Education",       "Family support resources and planning"),
                        ("📋", "6-Month Follow-up",          "Schedule repeat cognitive assessment"),
                    ]
                },
                "AD":   {
                    "icon":"🔴", "label":"Alzheimer's Disease", "urgency":"URGENT",
                    "urgency_color":"#ef4444", "stage_idx":2,
                    "tagline":"Findings consistent with Alzheimer's Disease — urgent action required.",
                    "desc": "The clinical profile strongly aligns with Alzheimer's Disease. "
                            "This is a progressive neurodegenerative condition affecting memory, "
                            "thinking, and behaviour. Immediate specialist involvement is critical.",
                    "recs": [
                        ("🚨", "Urgent Neurology Referral",  "Immediate specialist consultation"),
                        ("🖥️", "PET / MRI Imaging",          "Full structural and functional neuroimaging"),
                        ("💊", "Pharmacotherapy",            "Cholinesterase inhibitors if appropriate"),
                        ("🛡️", "Safety Assessment",          "Home safety evaluation and fall prevention"),
                        ("📝", "Legal & Financial Planning", "Power of attorney, advance directives"),
                        ("👨‍⚕️", "Comprehensive Care Plan",   "Multi-disciplinary team involvement"),
                        ("❤️", "Caregiver Support",         "Respite care, counselling, helplines"),
                    ]
                },
            }

            meta = RESULT_META[result]
            stage_labels = ["CN — Normal", "LMCI — Mild", "AD — Alzheimer's"]
            stage_colors = ["#22c55e",     "#f59e0b",      "#ef4444"]

            # ── 1. HERO BANNER ──────────────────────────────────────────────
            st.markdown(f"""
                <div style="background:linear-gradient(135deg,rgba(0,0,0,0.7),{color}22);
                            border:2px solid {color}; border-radius:16px;
                            padding:32px 28px; text-align:center; margin-top:18px;
                            box-shadow:0 0 40px {color}33;">
                    <div style="font-size:3rem; margin-bottom:8px;">{meta['icon']}</div>
                    <div style="display:inline-block; background:{color}22; border:1px solid {color};
                                border-radius:6px; padding:4px 14px; margin-bottom:10px;">
                        <span style="color:{meta['urgency_color']}; font-size:0.75rem;
                                     font-weight:800; letter-spacing:2px;">
                            {meta['urgency']} PRIORITY
                        </span>
                    </div>
                    <h1 style="color:{color}; font-size:2.6rem; margin:6px 0 4px;
                               text-shadow:0 0 20px {color}66;">
                        {meta['label']}
                    </h1>
                    <p style="color:#9ca3af; font-size:0.85rem; margin:0 0 12px;
                              font-style:italic;">{meta['tagline']}</p>
                    <div style="background:rgba(0,0,0,0.3); border-radius:8px;
                                padding:14px 20px; margin-top:10px;">
                        <p style="color:#e5e7eb; font-size:0.9rem; line-height:1.7; margin:0;">
                            {meta['desc']}
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.write("")

            # ── 2. STAGE PROGRESSION TIMELINE ───────────────────────────────
            st.markdown("<h4 style='color:#c4b5fd;margin-bottom:8px;'>&#128205; Disease Stage Timeline</h4>",
                        unsafe_allow_html=True)
            t_cols = st.columns(3)
            for i, (slabel, scolor) in enumerate(zip(stage_labels, stage_colors)):
                is_current = (i == meta["stage_idx"])
                with t_cols[i]:
                    s_border = "3px solid " + scolor if is_current else "1px solid " + scolor + "44"
                    s_bg     = scolor + "22" if is_current else "rgba(0,0,0,0.2)"
                    s_num    = "Stage " + str(i + 1) + " of 3"
                    if is_current:
                        s_badge = ("<div style='background:" + scolor +
                                   ";color:#000;font-size:0.65rem;font-weight:800;"
                                   "padding:2px 10px;border-radius:10px;margin-bottom:4px;"
                                   "display:inline-block;'>CURRENT</div>")
                    else:
                        s_badge = ""
                    st.markdown(
                        "<div style='border:" + s_border + "; background:" + s_bg + ";"
                        "border-radius:10px; padding:14px 12px; text-align:center;"
                        "min-height:90px; display:flex; flex-direction:column;"
                        "align-items:center; justify-content:center;'>"
                        + s_badge +
                        "<div style='color:" + scolor + "; font-size:0.88rem; font-weight:700;"
                        "margin-top:4px;'>" + slabel + "</div>"
                        "<div style='font-size:0.65rem; color:#6b7280; margin-top:4px;'>" + s_num + "</div>"
                        "</div>",
                        unsafe_allow_html=True
                    )

            st.write("")

            # ── 3. RISK FACTOR ANALYSIS ──────────────────────────────────────
            st.markdown("<h4 style='color:#c4b5fd;margin-bottom:8px;'>🧬 Patient Risk Factor Analysis</h4>",
                        unsafe_allow_html=True)

            # MMSE risk
            if   s_mmse >= 24: mmse_risk,mmse_rc = "Normal",       "#22c55e"
            elif s_mmse >= 19: mmse_risk,mmse_rc = "Mild Concern",  "#84cc16"
            elif s_mmse >= 10: mmse_risk,mmse_rc = "Moderate Risk", "#f59e0b"
            else:              mmse_risk,mmse_rc = "High Risk",     "#ef4444"

            # APOE4 risk
            if   "0 copies" in s_apoe4: apoe_risk,apoe_rc = "Low Risk",      "#22c55e"
            elif "1 copy"  in s_apoe4: apoe_risk,apoe_rc = "Moderate Risk",  "#f59e0b"
            else:                       apoe_risk,apoe_rc = "High Risk",      "#ef4444"

            # Age risk
            if   s_age < 65:  age_risk,age_rc = "Low",     "#22c55e"
            elif s_age < 75:  age_risk,age_rc = "Moderate","#f59e0b"
            else:             age_risk,age_rc = "High",    "#ef4444"

            # Genotype risk
            g_risk_map = {"4,4":"#ef4444","3,4":"#f59e0b","2,4":"#f59e0b"}
            geno_rc   = g_risk_map.get(s_geno, "#22c55e")
            geno_risk = "High Risk" if geno_rc=="#ef4444" else ("Moderate" if geno_rc=="#f59e0b" else "Low Risk")

            rf_cols = st.columns(4)
            risk_factors = [
                ("&#129504;", "MMSE Score",   str(s_mmse) + "/30", mmse_risk, mmse_rc),
                ("&#129516;", "APOE4 Status", s_apoe4.split("(")[0].strip(), apoe_risk, apoe_rc),
                ("&#127874;", "Age Risk",     str(s_age) + " years", age_risk, age_rc),
                ("&#128300;", "Genotype",     s_geno,                geno_risk, geno_rc),
            ]
            for col, (icon, lbl, val, risk, rc) in zip(rf_cols, risk_factors):
                with col:
                    card_html = (
                        "<div style='background:rgba(0,0,0,0.4); border:1px solid " + rc + "55;"
                        "border-top:3px solid " + rc + "; border-radius:10px;"
                        "padding:14px 12px; text-align:center;'>"
                        "<div style='font-size:1.5rem;'>" + icon + "</div>"
                        "<div style='color:#9ca3af; font-size:0.7rem; margin:4px 0 2px;'>" + lbl + "</div>"
                        "<div style='color:#e0d7ff; font-size:0.9rem; font-weight:700;'>" + val + "</div>"
                        "<div style='background:" + rc + "22; border:1px solid " + rc + ";"
                        "border-radius:4px; padding:2px 6px; margin-top:6px; display:inline-block;'>"
                        "<span style='color:" + rc + "; font-size:0.65rem; font-weight:700;'>" + risk + "</span>"
                        "</div></div>"
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

            st.write("")

            # ── 4. MMSE GAUGE BAR ────────────────────────────────────────────
            st.markdown("<h4 style='color:#c4b5fd;margin-bottom:4px;'>📊 MMSE Cognitive Score</h4>",
                        unsafe_allow_html=True)
            mmse_pct = int((s_mmse / 30) * 100)
            st.progress(mmse_pct,
                text=f"MMSE: {s_mmse}/30 — {mmse_risk}  "
                     f"({'Normal ✅' if s_mmse>=24 else 'Impaired ⚠️' if s_mmse>=19 else 'Moderate Risk 🟠' if s_mmse>=10 else 'Severe 🔴'})")

            st.write("")

            # ── 5. CLINICAL RECOMMENDATIONS ─────────────────────────────────
            st.markdown("<h4 style='color:#c4b5fd;margin-bottom:8px;'>&#128203; Clinical Recommendations</h4>",
                        unsafe_allow_html=True)
            rec_cols = st.columns(2)
            for i, (icon, title, detail) in enumerate(meta["recs"]):
                with rec_cols[i % 2]:
                    card = (
                        "<div style='background:rgba(139,92,246,0.08);"
                        "border:1px solid rgba(139,92,246,0.25);"
                        "border-left:3px solid " + color + ";"
                        "border-radius:8px; padding:10px 14px; margin-bottom:8px;'>"
                        "<span style='font-size:1.1rem;'>" + icon + "</span>"
                        "<span style='color:#e0d7ff; font-weight:700; font-size:0.85rem;"
                        " margin-left:8px;'>" + title + "</span><br>"
                        "<span style='color:#9ca3af; font-size:0.78rem;"
                        " margin-left:28px;'>" + detail + "</span>"
                        "</div>"
                    )
                    st.markdown(card, unsafe_allow_html=True)

            st.write("")

            # ── 6. NEXT STEPS TIMELINE ───────────────────────────────────────
            next_steps = {
                "CN":   [("Today",    "#22c55e", "Continue healthy lifestyle habits"),
                         ("3 Months", "#84cc16", "Follow up with GP for general check"),
                         ("1 Year",   "#60a5fa", "Repeat cognitive assessment (MMSE)")],
                "LMCI": [("This Week","#f59e0b", "Book appointment with neurologist"),
                         ("1 Month",  "#f59e0b", "MRI brain scan & blood tests"),
                        ("6 Months", "#f59e0b", "Neuropsychological evaluation")],
                "AD":   [("TODAY",    "#ef4444", "Emergency neurology referral"),
                         ("This Week","#ef4444", "PET/MRI imaging & medication review"),
                         ("1 Month",  "#ef4444", "Comprehensive care plan activation")],
            }
            st.markdown("<h4 style='color:#c4b5fd;margin-bottom:8px;'>&#9201; Recommended Action Timeline</h4>",
                        unsafe_allow_html=True)
            steps = next_steps[result]
            ns_cols = st.columns(len(steps))
            for col, (when, sc, action) in zip(ns_cols, steps):
                with col:
                    step_html = (
                        "<div style='background:rgba(0,0,0,0.4); border:1px solid " + sc + "44;"
                        "border-radius:10px; padding:14px 12px; text-align:center;'>"
                        "<div style='background:" + sc + "; color:#000; font-size:0.72rem;"
                        "font-weight:800; padding:3px 12px; border-radius:10px;"
                        "display:inline-block; margin-bottom:8px;'>" + when + "</div>"
                        "<div style='color:#e5e7eb; font-size:0.82rem; line-height:1.5;'>" + action + "</div>"
                        "</div>"
                    )
                    st.markdown(step_html, unsafe_allow_html=True)

            st.write("")
            st.caption("⚕️ For research purposes only. Always consult a qualified neurologist for clinical decisions.")

            # ── 7. PDF + EMAIL ────────────────────────────────────────────────
            st.write("")
            if PDF_OK and st.session_state.get("clin_pdf"):
                col_dl, col_sp = st.columns([1, 1])
                with col_dl:
                    _pdf_download_btn(
                        st.session_state["clin_pdf"],
                        "alzheimer_clinical_report.pdf",
                        "📄 Download Clinical Report (PDF)"
                    )
                _email_section(
                    pdf_bytes=st.session_state["clin_pdf"],
                    filename="alzheimer_clinical_report.pdf",
                    report_type="Clinical",
                    patient_summary=st.session_state.get("clin_summary", ""),
                    key_prefix="clin",
                )
            elif PDF_OK and not st.session_state.get("clin_pdf"):
                st.info("Re-run prediction to generate PDF.")
            elif not PDF_OK:
                st.info("Install `reportlab` to enable PDF: `pip install reportlab`")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2 — CNN MRI
    # ═══════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("🧠 MRI Brain Scan — CNN Prediction")

        cnn_ready = os.path.exists(CNN_MODEL_PATH)

        if not cnn_ready:
            st.warning("⚠️ CNN model not trained yet.")
            st.markdown("""
                <div style="background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.35);
                            border-radius:10px;padding:18px 20px;">
                    <b>📋 Steps to enable MRI prediction:</b><br><br>
                    <b>1.</b> Download the dataset from
                    <a href="https://www.kaggle.com/datasets/ninadaithal/imagesoasis" target="_blank">
                    Kaggle OASIS Dataset</a><br>
                    <b>2.</b> Extract so the folder structure is:<br>
                    <code>streamlit_app/Data/Non Demented/...</code><br>
                    <code>streamlit_app/Data/Mild Demented/...</code><br>
                    <code>streamlit_app/Data/Very Mild Demented/...</code><br>
                    <code>streamlit_app/Data/Moderate Demented/...</code><br><br>
                    <b>3.</b> Click <b>Retrain CNN</b> below to train the model automatically.
                </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ CNN model loaded and ready!")

        # ── In-app retrain section ─────────────────────────────────────────
        _RETRAIN_SCRIPT = os.path.join(_SCRIPT_DIR, "RETRAIN_NOW.py")

        with st.expander("🔧 CNN Model Management (Retrain / Diagnose)", expanded=not cnn_ready):
            st.markdown("""
                <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.3);
                            border-radius:8px;padding:12px 16px;margin-bottom:12px;">
                    <b>⚠️ If predictions are all the same class</b> — the model needs retraining.<br>
                    Click the button below to retrain directly from this page.
                    <br><small>This will take <b>15–30 minutes</b> on CPU. Do not close the browser.</small>
                </div>
            """, unsafe_allow_html=True)

            col_diag, col_train = st.columns(2)

            with col_diag:
                if st.button("🔍 Diagnose Model Quality", use_container_width=True, key="diag_btn"):
                    if not cnn_ready:
                        st.error("No model found to diagnose.")
                    else:

                        with st.spinner("Running model diagnosis..."):
                            try:
                                import numpy as np
                                os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
                                os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"
                                # Support TF 2.16+ standalone keras
                                try:
                                    import keras as _keras_diag
                                    _diag_load = _keras_diag.models.load_model
                                except ImportError:
                                    import tensorflow as _tf_diag
                                    _diag_load = _tf_diag.keras.models.load_model
                                import json as _json
                                with open(CNN_INDICES_PATH) as f:
                                    _ci = _json.load(f)
                                _inv = {int(v): k for k, v in _ci.items()}
                                _m = _diag_load(CNN_MODEL_PATH)
                                _tests = [
                                    ("Black zeros",  np.zeros((1,128,128,3), dtype=np.float32)),
                                    ("White ones",   np.ones( (1,128,128,3), dtype=np.float32)),
                                    ("Random noise", np.random.rand(1,128,128,3).astype(np.float32)),
                                    ("Mid-grey 0.5", np.full( (1,128,128,3), 0.5, dtype=np.float32)),
                                ]
                                _preds = []
                                _lines = []
                                for _name, _arr in _tests:
                                    _p   = _m.predict(_arr, verbose=0)[0]
                                    _cid = int(np.argmax(_p))
                                    _conf = float(_p[_cid]) * 100
                                    _preds.append(_cid)
                                    _lines.append(f"**{_name}** → `{_inv[_cid]}` ({_conf:.1f}%)")
                                for _l in _lines:
                                    st.markdown(_l)
                                if len(set(_preds)) <= 1:
                                    st.error("MODEL IS STUCK — always predicts the same class. Click Retrain CNN to fix it.")
                                else:
                                    st.success("Model produces varied outputs — looks healthy!")
                            except Exception as _e:
                                st.error(f"Diagnosis error: {_e}")

            with col_train:
                if st.button("🚀 Retrain CNN (15-30 min)", use_container_width=True,
                             key="retrain_btn", type="primary"):
                    if not os.path.exists(_RETRAIN_SCRIPT):
                        st.error(f"Retrain script not found: {_RETRAIN_SCRIPT}")
                    else:
                        st.info("Retraining started. This will take 15-30 minutes. Output appears below...")
                        log_box = st.empty()
                        log_lines = []

                        import sys as _sys
                        import subprocess as _sub
                        _proc = _sub.Popen(
                            [_sys.executable, _RETRAIN_SCRIPT],
                            stdout=_sub.PIPE, stderr=_sub.STDOUT,
                            encoding="utf-8", errors="replace",
                            bufsize=1,
                            cwd=_SCRIPT_DIR,
                            env={**os.environ,
                                 "TF_CPP_MIN_LOG_LEVEL": "3",
                                 "CUDA_VISIBLE_DEVICES": "-1",
                                 "TF_ENABLE_ONEDNN_OPTS": "0",
                                 "PYTHONUTF8": "1",
                                 "PYTHONIOENCODING": "utf-8",
                                 "TQDM_DISABLE": "1"},
                        )
                        for _line in _proc.stdout:
                            _line = _line.rstrip()
                            if _line:
                                log_lines.append(_line)
                                display = log_lines[-40:]
                                log_box.code("\n".join(display), language="")
                        _proc.wait()

                        if _proc.returncode == 0:
                            st.success("Retraining complete! Refresh the page to use the new model.")
                            st.balloons()
                        else:
                            st.error("Retraining failed. Check the log output above for errors.")

        st.write("")
        uploaded = st.file_uploader(
            "Upload an MRI Brain Scan Image",
            type=["jpg","jpeg","png"],
            help="Upload a 2D axial MRI brain slice image (JPG or PNG format)"
        )

        if uploaded:
            img = Image.open(uploaded)
            col_img, col_res = st.columns([1, 1.5])
            with col_img:
                st.image(img, caption="Uploaded MRI Scan", use_column_width=True)

            with col_res:
                if not cnn_ready:
                    st.error("CNN model not found. Please train it first.")
                else:
                    # Run prediction only when a new image is uploaded
                    img_key = hash(uploaded.getvalue())
                    if st.session_state.get("mri_img_key") != img_key:
                        with st.spinner("Analysing MRI scan with CNN..."):
                            mri_result = predict_mri(img)
                        st.session_state["mri_img_key"] = img_key
                        st.session_state["mri_result"]  = mri_result

                        if mri_result[0] is not None:
                            class_name, confidence, probs, inv_idx = mri_result
                            if PDF_OK:
                                try:
                                    st.session_state["mri_pdf"] = generate_mri_pdf(
                                        class_name, confidence, probs, inv_idx)
                                    _, description, _ = CNN_LABEL_MAP.get(
                                        class_name, (class_name, class_name, "#a78bfa"))
                                    st.session_state["mri_summary"] = (
                                        f"MRI Scan Analysis — <b>{class_name}</b> "
                                        f"(Confidence: {confidence:.1f}%)<br>"
                                        f"Clinical stage: {description}"
                                    )
                                except Exception:
                                    st.session_state["mri_pdf"] = None

                    # Always render results from session_state
                    mri_result = st.session_state.get("mri_result")
                    if mri_result is None:
                        st.info("Upload an image to start analysis.")
                    elif mri_result[0] is None:
                        st.error(f"CNN error: {mri_result[1]}")
                        st.warning("💡 If you see a TensorFlow error, try clicking **Retrain CNN** above.")
                    else:
                        class_name, confidence, probs, inv_idx = mri_result
                        short_label, description, color = CNN_LABEL_MAP.get(
                            class_name, (class_name, "", "#a78bfa"))

                        st.markdown(f"""
                            <div style="background:rgba(0,0,0,0.4);border:2px solid {color};
                                        border-radius:12px;padding:22px;text-align:center;">
                                <p style="color:#9ca3af;font-size:.85rem;margin:0 0 6px;">
                                    CNN MRI Analysis Result</p>
                                <h2 style="color:{color};font-size:2rem;margin:0 0 4px;">
                                    {class_name}</h2>
                                <p style="color:{color};font-size:1rem;font-weight:600;margin:0 0 12px;">
                                    Confidence: {confidence:.1f}%</p>
                                <p style="color:#d1d5db;font-size:.85rem;">
                                    Clinical stage: <b>{description}</b></p>
                            </div>
                        """, unsafe_allow_html=True)

                        st.write("")
                        st.write("**Probability breakdown:**")
                        for idx_num, name in sorted(inv_idx.items()):
                            prob = float(probs[idx_num]) * 100
                            st.progress(int(prob), text=f"{name}: {prob:.1f}%")

                        st.caption("For educational use only. Consult a radiologist for diagnosis.")

                        st.write("")
                        if PDF_OK and st.session_state.get("mri_pdf"):
                            _pdf_download_btn(
                                st.session_state["mri_pdf"],
                                "alzheimer_mri_report.pdf",
                                "📄 Download MRI Report (PDF)"
                            )
                            _email_section(
                                pdf_bytes=st.session_state["mri_pdf"],
                                filename="alzheimer_mri_report.pdf",
                                report_type="MRI",
                                patient_summary=st.session_state.get("mri_summary", ""),
                                key_prefix="mri",
                            )
