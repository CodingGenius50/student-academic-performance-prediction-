import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Student Academic Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root {
    --navy: #172B4D;
    --blue: #2563EB;
    --teal: #0D9488;
    --ink: #172033;
    --muted: #64748B;
    --canvas: #F5F7FB;
    --line: #E2E8F0;
}
/* Main canvas and typography */
.stApp, [data-testid="stAppViewContainer"] {
    background: var(--canvas) !important;
    color: var(--ink) !important;
}
[data-testid="stHeader"] {
    background: rgba(245,247,251,0.96) !important;
}
.block-container {
    max-width: 1440px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3, h4, p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
    color: var(--ink);
}
h1 {font-weight: 800; letter-spacing: -0.035em;}
h2, h3 {font-weight: 700; letter-spacing: -0.02em;}
[data-testid="stCaptionContainer"] p {color: var(--muted) !important;}
/* Navy sidebar */
section[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: 1px solid #243B60;
}
section[data-testid="stSidebar"] > div {
    background: var(--navy) !important;
}
section[data-testid="stSidebar"] * {
    color: #E8EEF8 !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #AFC1DA !important;
}
section[data-testid="stSidebar"] [role="radiogroup"] label {
    background: transparent;
    border-radius: 9px;
    padding: 7px 10px;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #243F68;
}
/* Hero */
.hero {
    padding: 34px 38px;
    border-radius: 20px;
    color: white;
    background: linear-gradient(120deg, #172B4D 0%, #1D4F91 58%, #2563EB 100%);
    margin: 0 0 24px 0;
    box-shadow: 0 12px 30px rgba(23,43,77,.14);
}
.hero h1 {
    color: #FFFFFF !important;
    font-size: clamp(2rem, 3.2vw, 2.8rem);
    line-height: 1.15;
    margin: 0 0 10px 0;
}
.hero p {color: #E6F0FF !important; font-size: 1.06rem; margin: 0;}
/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    padding: 20px 22px !important;
    border-radius: 16px !important;
    border: 1px solid var(--line) !important;
    box-shadow: 0 5px 16px rgba(15,23,42,.045);
    min-height: 112px;
}
[data-testid="stMetricLabel"] p {
    color: #64748B !important;
    font-weight: 600 !important;
    font-size: .88rem !important;
}
[data-testid="stMetricValue"] {
    color: var(--navy) !important;
    font-weight: 750 !important;
}
/* Inputs and buttons */
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border-color: #CBD5E1 !important;
    border-radius: 10px !important;
}
.stButton button, .stFormSubmitButton button, button[kind="primary"] {
    background: var(--blue) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--blue) !important;
    border-radius: 10px !important;
    font-weight: 650 !important;
    min-height: 42px;
}
.stButton button:hover, .stFormSubmitButton button:hover {
    background: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
}
/* Dataframes, tabs, expanders */
[data-testid="stDataFrame"], [data-testid="stTable"] {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 12px;
}
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 12px;
}
.stTabs [data-baseweb="tab-list"] {gap: 8px;}
.stTabs [data-baseweb="tab"] {
    background: #EAF0FA;
    border-radius: 9px 9px 0 0;
    padding: 9px 16px;
}
.stTabs [aria-selected="true"] {
    color: var(--blue) !important;
    border-bottom-color: var(--blue) !important;
}
.note {
    padding: 16px 18px;
    border-radius: 12px;
    background: #FFF8E7;
    border: 1px solid #F3D99A;
    border-left: 4px solid #D69E22;
    color: #51411D !important;
    line-height: 1.6;
}
div[data-testid="stAlert"] {border-radius: 12px;}
hr {border-color: var(--line) !important;}
/* Refined surfaces and clearer visual hierarchy */
.stApp, [data-testid="stAppViewContainer"] {
    background: #F1F5FA !important;
}
.block-container {padding-top: 2.2rem; padding-left: 2.5rem; padding-right: 2.5rem;}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #D8E2EF;
    border-radius: 16px;
    box-shadow: 0 8px 22px rgba(23,43,77,.055);
}
[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #D8E2EF;
    border-radius: 18px;
    padding: 24px 26px 18px 26px;
    box-shadow: 0 8px 24px rgba(23,43,77,.055);
}
[data-testid="stForm"] h4 {
    color: #1D4F91 !important;
    letter-spacing: -.01em;
}
[data-testid="stForm"] [data-testid="stCaptionContainer"] p {
    color: #64748B !important;
}
[data-testid="stForm"] label {
    color: #26364D !important;
    font-weight: 600 !important;
}
[data-testid="stForm"] [data-baseweb="select"] > div,
[data-testid="stForm"] input {
    background: #F8FAFD !important;
    border: 1px solid #CBD7E6 !important;
    border-radius: 10px !important;
    min-height: 44px;
}
.info-banner {
    background: #EAF2FF;
    border: 1px solid #C9DBF8;
    border-left: 4px solid #2563EB;
    color: #24436D;
    padding: 14px 18px;
    border-radius: 12px;
    margin: 16px 0 20px 0;
    line-height: 1.55;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #FFFFFF 0%, #F6F9FF 100%) !important;
    border: 1px solid #D8E2EF !important;
}
.stProgress > div > div > div > div {background: #0D9488;}

</style>
""", unsafe_allow_html=True)

ASSET = "assets"
MODEL_PATH = os.path.join(ASSET, "best_student_model.pkl")
DATA_PATH = os.path.join(ASSET, "student_data.csv")
RESULTS_PATH = os.path.join(ASSET, "model_comparison.csv")
IMPORTANCE_PATH = os.path.join(ASSET, "feature_importance.csv")
COLUMNS_PATH = os.path.join(ASSET, "model_columns.pkl")

required = [MODEL_PATH, DATA_PATH, RESULTS_PATH, IMPORTANCE_PATH, COLUMNS_PATH]
missing = [p for p in required if not os.path.exists(p)]
if missing:
    st.error("Required project files are missing: " + ", ".join(missing))
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_PATH)

@st.cache_data
def load_importance():
    return pd.read_csv(IMPORTANCE_PATH)

model = load_model()
data = load_data()
results = load_results()
importance = load_importance()
feature_columns = joblib.load(COLUMNS_PATH)

st.sidebar.title("🎓 Student Performance")
st.sidebar.caption("Machine Learning Project")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Dataset Explorer", "Exploratory Data Analysis",
     "Model Comparison", "Feature Analysis", "Live Prediction", "About Project"]
)
st.sidebar.divider()
st.sidebar.caption("UCI Student Performance — Mathematics dataset")

if page == "Dashboard":
    st.markdown("""
    <div class="hero">
      <h1>Student Academic Performance Prediction</h1>
      <p>Predicting students' final mathematics grade using selected study and personal factors.</p>
    </div>
    """, unsafe_allow_html=True)

    best_row = results.sort_values("RMSE").iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", f"{len(data):,}")
    c2.metric("Input Features", len(feature_columns))
    c3.metric("Best Test RMSE", f"{best_row['RMSE']:.3f}")
    c4.metric("Best Model", str(best_row["Model"]))

    st.subheader("Project Overview")
    left, right = st.columns([1.2, 1])
    with left:
        st.write(
            "This project uses a regression model to estimate the final mathematics "
            "grade (G3) on a 0–20 scale. The prediction uses eight inputs and does "
            "not use the earlier-period grades G1 or G2."
        )
        st.markdown("""
        **Selected inputs**
        - Study time category and age
        - Absences and previous failures
        - Free-time category and health category
        - Family educational support
        - Paid extra classes
        """)
    with right:
        st.subheader("Target grade distribution")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.histplot(data=data, x="G3", bins=20, kde=True, ax=ax, color="#2878b5")
        ax.set_xlabel("Final grade (G3)")
        ax.set_ylabel("Number of students")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown(
        '<div class="note"><b>Model performance note:</b> The current best test R² is '
        f'{best_row["R2 Score"]:.3f}. A negative R² means this model did not outperform '
        'a simple test-set mean baseline on this split. Treat predictions as estimates, '
        'not guaranteed results.</div>', unsafe_allow_html=True
    )

elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(data))
    c2.metric("Columns", len(data.columns))
    c3.metric("Missing Values", int(data.isna().sum().sum()))
    st.dataframe(data, use_container_width=True, height=440)
    st.download_button(
        "Download selected dataset (CSV)",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="student_selected_dataset.csv",
        mime="text/csv"
    )
    with st.expander("Dataset summary"):
        st.dataframe(data.describe(include="all").T, use_container_width=True)

elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.write("Explore the distribution of final grades and selected feature relationships.")
    tab1, tab2, tab3 = st.tabs(["Grade Distribution", "Categorical Features", "Absences"])
    with tab1:
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(data=data, x="G3", bins=20, kde=True, ax=ax, color="#2878b5")
        ax.set_title("Distribution of Final Mathematics Grades (G3)")
        ax.set_xlabel("Final grade (0–20)")
        ax.set_ylabel("Number of students")
        st.pyplot(fig)
        plt.close(fig)
    with tab2:
        choices = ["studytime", "failures", "freetime", "health", "famsup", "paid"]
        chosen = st.selectbox("Select a feature", choices)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=data, x=chosen, y="G3", ax=ax, color="#8db9df")
        ax.set_title(f"{chosen} vs final grade")
        st.pyplot(fig)
        plt.close(fig)
    with tab3:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(data=data, x="absences", y="G3", alpha=.7, ax=ax, color="#2878b5")
        ax.set_title("Absences vs Final Grade")
        st.pyplot(fig)
        plt.close(fig)

elif page == "Model Comparison":
    st.title("Model Performance Comparison")
    st.caption("Test-set metrics from the current 80/20 split (random_state=42).")
    display_results = results.sort_values("RMSE").reset_index(drop=True)
    st.dataframe(display_results, use_container_width=True, hide_index=True)
    metric = st.selectbox("Visualize metric", ["RMSE", "MAE", "R2 Score"])
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=results, x="Model", y=metric, ax=ax, color="#4b8fc5")
    ax.set_title(f"Model comparison — {metric}")
    ax.tick_params(axis="x", rotation=12)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown(
        '<div class="note">Lower MAE and RMSE are better. Higher R² is better. '
        'A negative R² indicates performance worse than predicting the test-set mean.</div>',
        unsafe_allow_html=True
    )

elif page == "Feature Analysis":
    st.title("Feature Analysis")
    st.write("The supplied file stores absolute regression coefficients for the selected Linear Regression model. "
             "These are coefficient magnitudes, not normalized feature-importance percentages.")
    if "Feature" in importance.columns and "Importance" in importance.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_data = importance.sort_values("Importance", ascending=True)
        sns.barplot(data=plot_data, x="Importance", y="Feature", ax=ax, color="#4b8fc5")
        ax.set_title("Absolute Linear Regression Coefficient Magnitudes")
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(importance.sort_values("Importance", ascending=False),
                     use_container_width=True, hide_index=True)
    else:
        st.warning("Feature analysis file does not contain the expected columns.")

elif page == "Live Prediction":
    st.title("Live Student Performance Prediction")
    st.write(
        "Estimate a student's final mathematics grade (G3) from study habits, "
        "attendance, previous failures and selected personal factors."
    )
    st.markdown(
        '<div class="info-banner"><b>How to read the inputs:</b> '
        'Some fields are categories, not physical measurements. Each category is '
        'explained beside the input so you can select the correct value.</div>',
        unsafe_allow_html=True
    )

    study_labels = {
        1: "1 — Less than 2 hours per week",
        2: "2 — 2 to 5 hours per week",
        3: "3 — 5 to 10 hours per week",
        4: "4 — More than 10 hours per week"
    }
    free_labels = {
        1: "1 — Very low",
        2: "2 — Low",
        3: "3 — Medium",
        4: "4 — High",
        5: "5 — Very high"
    }
    health_labels = {
        1: "1 — Very bad",
        2: "2 — Bad",
        3: "3 — Fair",
        4: "4 — Good",
        5: "5 — Very good"
    }

    with st.form("prediction_form"):
        st.markdown("#### 01 · Study habits and attendance")
        st.caption("Enter the student's study-time category, age and recorded absences.")
        left, right = st.columns(2)
        with left:
            studytime = st.selectbox(
                "Weekly study time",
                options=[1, 2, 3, 4],
                index=1,
                format_func=lambda value: study_labels[value],
                help="Choose the category that best describes total weekly study time. The number is a dataset code, not the number of hours."
            )
            age = st.number_input(
                "Student age (years)",
                min_value=15, max_value=22, value=17, step=1,
                help="Age in completed years."
            )
        with right:
            absences = st.number_input(
                "School absences (count)",
                min_value=0, max_value=75, value=2, step=1,
                help="Number of recorded school absences in the source dataset."
            )
            failures = st.selectbox(
                "Previous class failures (count)",
                options=[0, 1, 2, 3],
                help="Number of past class failures recorded in the dataset."
            )

        st.markdown("---")
        st.markdown("#### 02 · Personal and support factors")
        st.caption("These are ordinal categories or yes/no indicators—not measured units.")
        left, right = st.columns(2)
        with left:
            freetime = st.selectbox(
                "Free-time level (category)",
                options=[1, 2, 3, 4, 5],
                index=2,
                format_func=lambda value: free_labels[value],
                help="Self-reported free time after school, from very low to very high."
            )
            famsup = st.selectbox(
                "Family educational support",
                options=["yes", "no"],
                format_func=lambda value: "Yes" if value == "yes" else "No",
                help="Whether the student receives family educational support."
            )
        with right:
            paid = st.selectbox(
                "Paid extra classes",
                options=["yes", "no"],
                format_func=lambda value: "Yes" if value == "yes" else "No",
                help="Whether the student takes paid extra educational classes."
            )
            health = st.selectbox(
                "Self-reported health (category)",
                options=[1, 2, 3, 4, 5],
                index=3,
                format_func=lambda value: health_labels[value],
                help="Self-reported health rating: 1 is very bad and 5 is very good."
            )

        submitted = st.form_submit_button(
            "Estimate Final Mathematics Grade",
            type="primary",
            use_container_width=True
        )

    if submitted:
        row = pd.DataFrame([{
            "studytime": studytime,
            "age": age,
            "absences": absences,
            "failures": failures,
            "freetime": freetime,
            "famsup": 1 if famsup == "yes" else 0,
            "paid": 1 if paid == "yes" else 0,
            "health": health
        }])
        row = row[feature_columns]
        prediction = float(model.predict(row)[0])
        prediction = float(np.clip(prediction, 0, 20))

        st.markdown("### Prediction result")
        with st.container(border=True):
            result_col, scale_col = st.columns([1, 2])
            with result_col:
                st.metric("Estimated final grade (G3)", f"{prediction:.2f} / 20")
            with scale_col:
                st.markdown("**Position on the 0–20 grade scale**")
                st.progress(int(round(prediction / 20 * 100)))
                st.caption(f"Estimated score: {prediction:.2f} out of 20")
            st.caption(
                "This is a statistical estimate from the trained model, not an official "
                "grade or a guarantee of future academic performance."
            )
        with st.expander("Review submitted information"):
            readable_inputs = pd.DataFrame([{
                "Weekly study time": study_labels[studytime],
                "Age": f"{age} years",
                "School absences": f"{absences} recorded absences",
                "Previous class failures": failures,
                "Free-time level": free_labels[freetime],
                "Family educational support": famsup.title(),
                "Paid extra classes": paid.title(),
                "Self-reported health": health_labels[health]
            }])
            st.dataframe(readable_inputs, use_container_width=True, hide_index=True)

elif page == "About Project":
    st.title("About the Project")
    st.subheader("Objective")
    st.write("To build a regression-based system that estimates a student's final mathematics grade (G3) "
             "using eight selected features.")
    st.subheader("Dataset")
    st.write("The project uses the Mathematics subset of the UCI Student Performance dataset. "
             "The dataset contains 395 student records and 33 original columns; this project uses "
             "eight input features plus G3 as the target.")
    st.markdown("""
    **Target variable:** `G3` — final mathematics grade, 0–20.

    **Excluded variables:** `G1` and `G2` are not used as inputs, to avoid relying on earlier-period grades.

    **Methods:** exploratory data analysis, binary encoding for yes/no inputs, 80/20 train-test split,
    and comparison of Linear Regression, Decision Tree and Random Forest regression models.
    """)
    st.subheader("Limitations")
    st.markdown("""
    - The dataset is relatively small (395 records).
    - The data represents a specific Portuguese school context and is not a Bangladesh-wide student sample.
    - Predictive relationships do not establish cause and effect.
    - Current test-set metrics indicate limited predictive performance; improve validation and feature selection before real-world use.
    - Predictions should be treated as estimates, not decisions about students.
    """)
    st.subheader("Data source")
    st.markdown("- [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)")
    st.markdown("- [Mendeley dataset copy (DOI: 10.17632/65yg895xwb.1)](https://data.mendeley.com/datasets/65yg895xwb/1)")
    st.caption("Academic project prototype.")