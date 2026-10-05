import streamlit as st
import pandas as pd
import joblib
import os
@st.cache_resource
def load_models():
    model = joblib.load("model/fraud_detection_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

model, scaler = load_models()

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
# =========================================================
# RESPONSIVE CUSTOM CSS
# =========================================================
st.markdown("""
<style>

/* ---------- GENERAL ---------- */

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(56,189,248,0.10), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(139,92,246,0.10), transparent 25%),
        linear-gradient(135deg, #07111f 0%, #0b1628 50%, #050b14 100%);
    color: #f8fafc;
}

/* ---------- TITLE ---------- */

.hero-title {
    font-size: clamp(30px, 5vw, 48px);
    font-weight: 800;
    margin-bottom: 5px;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: clamp(13px, 2vw, 17px);
    margin-bottom: 25px;
}

/* ---------- CARDS ---------- */

.card {
    background: rgba(15,23,42,0.72);
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.25);
    margin-bottom: 18px;
    overflow-wrap: anywhere;
}

.metric-card {
    background: linear-gradient(
        145deg,
        rgba(30,41,59,0.9),
        rgba(15,23,42,0.75)
    );
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    min-height: 120px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
}

.metric-icon {
    font-size: 26px;
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 7px;
}

.metric-value {
    font-size: clamp(22px, 4vw, 30px);
    font-weight: 800;
    color: #f8fafc;
    margin-top: 5px;
}

/* ---------- SECTION TITLES ---------- */

.section-title {
    font-size: clamp(19px, 3vw, 24px);
    font-weight: 700;
    margin-top: 24px;
    margin-bottom: 14px;
    color: #e2e8f0;
}

/* ---------- ALERTS ---------- */

.fraud-alert,
.safe-alert {
    border-radius: 15px;
    padding: 16px;
    margin: 15px 0;
    overflow-wrap: anywhere;
}

.fraud-alert {
    background: rgba(127,29,29,0.35);
    border: 1px solid rgba(248,113,113,0.45);
    color: #fecaca;
}

.safe-alert {
    background: rgba(20,83,45,0.35);
    border: 1px solid rgba(74,222,128,0.4);
    color: #bbf7d0;
}

/* ---------- RISK BADGES ---------- */

.high-risk,
.medium-risk,
.low-risk {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 20px;
    font-weight: 700;
    margin: 8px 0;
}

.high-risk {
    background: rgba(239,68,68,0.18);
    color: #fca5a5;
}

.medium-risk {
    background: rgba(245,158,11,0.18);
    color: #fcd34d;
}

.low-risk {
    background: rgba(34,197,94,0.18);
    color: #86efac;
}

/* ---------- BUTTONS ---------- */

.stButton > button,
.stDownloadButton > button {
    width: 100%;
    border-radius: 10px;
    border: 1px solid rgba(56,189,248,0.3);
    background: linear-gradient(90deg, #0284c7, #4f46e5);
    color: white;
    font-weight: 700;
    padding: 10px 16px;
}

/* ---------- FILE UPLOADER ---------- */

section[data-testid="stFileUploader"] {
    background: rgba(15,23,42,0.65);
    border-radius: 16px;
    padding: 10px;
}

/* ---------- DATAFRAME ---------- */

[data-testid="stDataFrame"] {
    width: 100%;
}

/* ---------- SIDEBAR ---------- */

section[data-testid="stSidebar"] {
    background: rgba(5,12,24,0.97);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* ---------- TABS ---------- */

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    flex-wrap: wrap;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 9px 15px;
}

/* ---------- FOOTER ---------- */

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding: 30px 10px 10px;
    line-height: 1.6;
}


/* =========================================================
   MOBILE DEVICES
   ========================================================= */

@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1rem !important;
    }

    .hero-title {
        font-size: 32px;
    }

    .hero-subtitle {
        font-size: 14px;
    }

    .card {
        padding: 16px;
        border-radius: 14px;
    }

    .metric-card {
        min-height: 105px;
        padding: 14px 8px;
    }

    .metric-icon {
        font-size: 22px;
    }

    .metric-label {
        font-size: 11px;
    }

    .metric-value {
        font-size: 20px;
    }

    .section-title {
        font-size: 19px;
    }

    .stButton > button,
    .stDownloadButton > button {
        font-size: 14px;
        padding: 9px;
    }

    .footer {
        font-size: 11px;
    }
}


/* =========================================================
   VERY SMALL PHONES
   ========================================================= */

@media (max-width: 480px) {

    .block-container {
        padding-left: 0.7rem !important;
        padding-right: 0.7rem !important;
    }

    .hero-title {
        font-size: 27px;
    }

    .hero-subtitle {
        font-size: 13px;
    }

    .metric-card {
        padding: 12px 5px;
    }

    .metric-label {
        font-size: 10px;
    }

    .metric-value {
        font-size: 18px;
    }

    .card {
        padding: 13px;
    }

    .fraud-alert,
    .safe-alert {
        padding: 12px;
        font-size: 13px;
    }
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
MODEL_PATH = "model/fraud_detection_model.pkl"
SCALER_PATH = "model/scaler.pkl"


if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("⚠️ Model files not found. Please run train_model.py first.")
    st.stop()


@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


model, scaler = load_models()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("## 💳 FraudShield AI")
    st.caption("Machine Learning Fraud Detection")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔍 Analyze Transactions",
            "📊 Model Performance"
        ]
    )

    st.markdown("---")

    st.markdown("### 🧠 Technology")

    st.write("🐍 Python")
    st.write("🤖 Scikit-learn")
    st.write("🌲 Random Forest")
    st.write("📊 Streamlit")

    st.markdown("---")

    st.caption("Credit Card Fraud Detection Project")


# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="hero-title">💳 FraudShield AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'Credit Card Fraud Detection Using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# DASHBOARD
# =========================================================
if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">🚀 Project Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
        <h3>🎯 Objective</h3>
        <p>
        Detect potentially fraudulent credit card transactions
        using machine learning techniques.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
        <h3>🌲 Best Model</h3>
        <p>
        Random Forest achieved the best overall F1-score
        among the tested models.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">📂 Quick Start</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Go to **🔍 Analyze Transactions** from the sidebar "
        "and upload your creditcard.csv file."
    )

    st.markdown(
        '<div class="section-title">🔄 System Workflow</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    workflow = [
        ("📁", "Upload Data"),
        ("⚙️", "Preprocess"),
        ("🤖", "Predict"),
        ("🚨", "Detect Fraud")
    ]

    for col, (icon, text) in zip([c1, c2, c3, c4], workflow):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-label">{text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# ANALYZE TRANSACTIONS
# =========================================================
elif page == "🔍 Analyze Transactions":

    st.markdown(
        '<div class="section-title">📁 Upload Transaction Dataset</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload creditcard.csv",
        type=["csv"]
    )

    if uploaded_file is None:

        st.info(
            "Upload your CSV dataset to start fraud detection."
        )

    else:

        df = pd.read_csv(uploaded_file)

        st.success(
            f"Dataset loaded successfully — {len(df):,} transactions found."
        )

        required_columns = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)
        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:

            st.error(
                "The uploaded file is missing required columns: "
                + ", ".join(missing_columns)
            )

            st.stop()

        # -------------------------------------------------
        # PREPARE DATA
        # -------------------------------------------------

        X = df[required_columns].copy()

        X[["Time", "Amount"]] = scaler.transform(
            X[["Time", "Amount"]]
        )

        predictions = model.predict(X)

        probabilities = model.predict_proba(X)[:, 1]

        result_df = df.copy()

        result_df["Prediction"] = predictions

        result_df["Fraud Probability"] = probabilities

        result_df["Risk Level"] = pd.cut(
            probabilities,
            bins=[-0.01, 0.30, 0.70, 1.01],
            labels=["Low", "Medium", "High"]
        )

        result_df["Result"] = result_df["Prediction"].map({
            0: "NORMAL",
            1: "FRAUDULENT"
        })

        # -------------------------------------------------
        # DASHBOARD METRICS
        # -------------------------------------------------

        total = len(result_df)

        normal_count = int(
            (result_df["Prediction"] == 0).sum()
        )

        fraud_count = int(
            (result_df["Prediction"] == 1).sum()
        )

        fraud_percentage = (
            fraud_count / total * 100
            if total > 0 else 0
        )

        st.markdown(
            '<div class="section-title">📊 Detection Dashboard</div>',
            unsafe_allow_html=True
        )

        m1, m2, m3, m4 = st.columns(4)

        metrics = [
            ("📦", "Total Transactions", f"{total:,}"),
            ("✅", "Normal", f"{normal_count:,}"),
            ("🚨", "Fraudulent", f"{fraud_count:,}"),
            ("📈", "Fraud Rate", f"{fraud_percentage:.2f}%")
        ]

        for col, (icon, label, value) in zip(
            [m1, m2, m3, m4],
            metrics
        ):

            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-icon">{icon}</div>
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # -------------------------------------------------
        # ALERT
        # -------------------------------------------------

        if fraud_count > 0:

            st.markdown(
                f"""
                <div class="fraud-alert">
                    🚨 <b>Fraud Alert:</b>
                    The model identified
                    <b>{fraud_count:,}</b>
                    potentially fraudulent transactions.
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="safe-alert">
                    ✅ <b>No Fraud Detected:</b>
                    No transactions were classified as fraudulent.
                </div>
                """,
                unsafe_allow_html=True
            )

        # -------------------------------------------------
        # CHART
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">📈 Prediction Summary</div>',
            unsafe_allow_html=True
        )

        chart_data = pd.DataFrame({
            "Transaction Type": ["Normal", "Fraudulent"],
            "Count": [normal_count, fraud_count]
        })

        st.bar_chart(
            chart_data.set_index("Transaction Type")
        )

        # -------------------------------------------------
        # TRANSACTION ANALYZER
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">🔎 Transaction Analyzer</div>',
            unsafe_allow_html=True
        )

        transaction_index = st.number_input(
            "Select transaction number",
            min_value=0,
            max_value=max(0, total - 1),
            value=0,
            step=1
        )

        selected = result_df.iloc[int(transaction_index)]

        a1, a2, a3 = st.columns(3)

        probability = float(
            selected["Fraud Probability"]
        )

        risk = str(
            selected["Risk Level"]
        )

        with a1:
            st.metric(
                "Transaction",
                f"#{int(transaction_index) + 1}"
            )

        with a2:
            st.metric(
                "Fraud Probability",
                f"{probability * 100:.2f}%"
            )

        with a3:
            st.metric(
                "Prediction",
                selected["Result"]
            )

        if risk == "High":

            st.markdown(
                '<span class="high-risk">🔴 HIGH RISK</span>',
                unsafe_allow_html=True
            )

        elif risk == "Medium":

            st.markdown(
                '<span class="medium-risk">🟡 MEDIUM RISK</span>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<span class="low-risk">🟢 LOW RISK</span>',
                unsafe_allow_html=True
            )

        # -------------------------------------------------
        # FILTER
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">🔍 Transaction Results</div>',
            unsafe_allow_html=True
        )

        show_only_fraud = st.checkbox(
            "🚨 Show only potentially fraudulent transactions"
        )

        if show_only_fraud:

            display_df = result_df[
                result_df["Prediction"] == 1
            ]

        else:

            display_df = result_df

        st.dataframe(
            display_df.head(100),
            use_container_width=True
        )

        st.caption(
            f"Showing up to 100 records out of "
            f"{len(display_df):,} matching transactions."
        )

        # -------------------------------------------------
        # DOWNLOAD
        # -------------------------------------------------

        csv_data = result_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Prediction Results",
            data=csv_data,
            file_name="fraud_detection_results.csv",
            mime="text/csv"
        )




# =========================================================
# MODEL PERFORMANCE
# =========================================================
elif page == "📊 Model Performance":

    st.markdown(
        '<div class="section-title">🤖 Model Evaluation Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">
        <h3>📊 Machine Learning Model Comparison</h3>
        <p>
        Three machine learning algorithms were evaluated using
        Accuracy, Precision, Recall and F1 Score.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Model results
    performance = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ],
        "Accuracy": [
            0.9755,
            0.9907,
            0.9995
        ],
        "Precision": [
            0.0609,
            0.1321,
            0.9059
        ],
        "Recall": [
            0.9184,
            0.7857,
            0.7857
        ],
        "F1 Score": [
            0.1141,
            0.2261,
            0.8415
        ]
    })

    # -----------------------------------------------------
    # METRIC CARDS
    # -----------------------------------------------------

    best_model = performance.loc[
        performance["F1 Score"].idxmax()
    ]

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        ("🏆", "Best Model", best_model["Model"]),
        ("🎯", "Accuracy", f"{best_model['Accuracy'] * 100:.2f}%"),
        ("🔎", "Precision", f"{best_model['Precision'] * 100:.2f}%"),
        ("⭐", "F1 Score", f"{best_model['F1 Score'] * 100:.2f}%")
    ]

    for col, (icon, label, value) in zip(
        [c1, c2, c3, c4],
        metrics
    ):

        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # -----------------------------------------------------
    # PERFORMANCE TABLE
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📋 Performance Comparison</div>',
        unsafe_allow_html=True
    )

    display_performance = performance.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]:
        display_performance[column] = (
            display_performance[column] * 100
        ).round(2).astype(str) + "%"

    st.dataframe(
        display_performance,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # CHART
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📈 Model Performance Chart</div>',
        unsafe_allow_html=True
    )

    chart_data = performance.set_index("Model")[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    ]

    st.bar_chart(chart_data)

    # -----------------------------------------------------
    # RANDOM FOREST
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🌲 Why Random Forest?</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">
        <h3>🏆 Random Forest is our selected model</h3>

        <p>
        Random Forest achieved the highest overall F1 Score
        among the three tested algorithms.
        </p>

        <ul>
            <li><b>Accuracy:</b> 99.95%</li>
            <li><b>Precision:</b> 90.59%</li>
            <li><b>Recall:</b> 78.57%</li>
            <li><b>F1 Score:</b> 84.15%</li>
        </ul>

        <p>
        Since credit card fraud datasets are highly imbalanced,
        F1 Score, Precision and Recall are important alongside
        accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🧩 Confusion Matrix</div>',
        unsafe_allow_html=True
    )

    st.info(
        "The confusion matrix shows how correctly the model "
        "distinguishes Normal and Fraudulent transactions."
    )

    cm1, cm2 = st.columns(2)

    with cm1:

        st.markdown("""
        <div class="card">
            <h3>True Negative (TN)</h3>
            <p>
            Normal transactions correctly classified as Normal.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <h3>False Positive (FP)</h3>
            <p>
            Normal transactions incorrectly classified as Fraud.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with cm2:

        st.markdown("""
        <div class="card">
            <h3>True Positive (TP)</h3>
            <p>
            Fraudulent transactions correctly detected as Fraud.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <h3>False Negative (FN)</h3>
            <p>
            Fraudulent transactions incorrectly classified as Normal.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.success(
        "💡 In fraud detection, reducing False Negatives is "
        "particularly important because missed fraud can be risky."
    )
# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        💳 FraudShield AI &nbsp;|&nbsp;
        Credit Card Fraud Detection Using Machine Learning
        <br>
        Built with Python • Scikit-learn • Random Forest • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)