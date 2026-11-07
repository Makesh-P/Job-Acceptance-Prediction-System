from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "job_acceptance_model.pkl"
DATA_PATH = BASE_DIR / "Cleaned_Dataset.csv"
IMAGE_DIR = BASE_DIR / "images"
STYLE_PATH = BASE_DIR / "style.css"

EDA_CHARTS = {
    "1. Academic scores vs placement": (
        "eda_1_academic.png",
        "The average academic scores are almost the same for both placed and not-placed candidates. This suggests that academic performance alone does not make a major difference in placement outcomes."
    ),

    "2. Skills match vs interview performance": (
        "eda_2_skills_interview.png",
        "Candidates with higher interview scores appear more frequently among the placed candidates. However, both groups overlap across different skill-match percentages, suggesting that skills match alone does not determine placement."
    ),

    "3. Certification impact on acceptance": (
        "eda_3_certifications.png",
        "Most candidates have an acceptance rate of around 30%, regardless of the number of certifications. Having more certifications does not show a consistent increase in acceptance rate."
    ),

    "4. Acceptance rate by company tier": (
        "eda_4_company_tier.png",
        "The acceptance rates are nearly the same across all three company tiers. This indicates that company tier does not show a significant difference in candidate acceptance rates."
    ),

    "5. Experience vs placement success": (
        "eda_5_experience.png",
        "Candidates with more experience have higher acceptance rates. Senior candidates show the highest rate, followed by juniors and freshers."
    ),

    "6. Interview score vs placement probability": (
    "eda_6_interview.png",
    "Candidates with higher interview scores generally receive higher placement probabilities. This shows a positive relationship between interview performance and predicted placement chances."
    ),

    "7. Aptitude score analysis": (
        "eda_7_aptitude.png",
        "Most candidates have aptitude scores between 55 and 70. The number of candidates decreases as the scores increase, and the distribution follows a similar pattern for both placed and not-placed candidates."
    )
}

def show_dashboard(model, threshold, features):

    df = pd.read_csv(DATA_PATH)

    total = len(df)
    placed = (df["status"] == "Placed").sum()

    interview_avg = df[
        ["technical_score", "aptitude_score", "communication_score"]
    ].mean(axis=1)

    probabilities = model.predict_proba(df[features])[:, 1]
    
    placement_rate = placed / total * 100
    acceptance_rate = probabilities.mean() * 100
    avg_interview = interview_avg.mean()
    avg_skills = df["skills_match_percentage"].mean()
    dropout_rate = (total - placed) / total * 100
    high_risk = (probabilities < threshold).mean() * 100

    st.subheader("Key Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Candidates", total)
    col2.metric("Placement Rate", f"{placement_rate:.1f}%")
    col3.metric("Job Acceptance Rate", f"{acceptance_rate:.1f}%")
    col4.metric("Average Interview Score", f"{avg_interview:.1f}")

    col5, col6, col7 = st.columns(3)
    col5.metric("Average Skills Match", f"{avg_skills:.1f}%")
    col6.metric("Offer Dropout Rate", f"{dropout_rate:.1f}%")
    col7.metric("High-Risk Candidates", f"{high_risk:.1f}%")

    st.divider()
    st.subheader("Exploratory Data Analysis")
    title = st.radio("Select a chart", list(EDA_CHARTS), horizontal=True, label_visibility="collapsed")
    filename, insight = EDA_CHARTS[title]

    chart_col, insight_col = st.columns([2, 1])

    with chart_col:
        if (IMAGE_DIR / filename).exists():
            st.image(str(IMAGE_DIR / filename), use_container_width=True)
        else:
            st.info(f"Add your chart image to images/{filename}")

    with insight_col:
        st.markdown(
            f'<div class="takeaway"><h4>Key Takeaway</h4>{insight}</div>',
            unsafe_allow_html=True
        )

def show_prediction(model, threshold, features):

    st.subheader("Candidate Application")
    df = pd.read_csv(DATA_PATH)

    with st.form("candidate_application"):

        with st.container(border=True):
            t1, t2, t3, t4, t5 = st.tabs([
                "1. Personal", "2. Skills", "3. Experience", "4. Job", "5. Preferences"
            ])

            with t1:
                st.markdown("### Personal & Education Details")
                a, b, c = st.columns(3)
                age = a.number_input("Age", min_value=18, max_value=65, value=26)
                gender = b.selectbox("Gender", sorted(df["gender"].dropna().unique()))
                degree = c.selectbox(
                    "Degree Specialization", sorted(df["degree_specialization"].dropna().unique())
                )
                a, b, c = st.columns(3)
                ssc = a.slider("SSC Percentage", 0.0, 100.0, 0.0, step=0.5)
                hsc = b.slider("HSC Percentage", 0.0, 100.0, 0.0, step=0.5)
                degree_percentage = c.slider("Degree Percentage", 0.0, 100.0, 0.0, step=0.5)

            with t2:
                st.markdown("### Skills & Interview Details")
                a, b, c = st.columns(3)
                technical = a.slider("Technical Score", min_value=0.0, max_value=100.0, step=0.5)
                aptitude = b.slider("Aptitude Score", min_value=0.0, max_value=100.0, value=0.0)
                communication = c.slider(
                    "Communication Score", min_value=0.0, max_value=100.0, value=0.0
                )
                a, b, c = st.columns(3)
                skills = a.slider(
                    "Skills Match Percentage", min_value=0.0, max_value=100.0, step=0.5
                )
                certifications = b.number_input(
                    "Certifications Count", min_value=0, max_value=20, value=0
                )
                internship = c.selectbox(
                    "Internship Experience", sorted(df["internship_experience"].dropna().unique())
                )

            with t3:
                st.markdown("### Experience & Employment Details")
                a, b, c = st.columns(3)
                experience = a.number_input(
                    "Years of Experience", min_value=0, max_value=20, value=0
                )
                relevant = b.selectbox(
                    "Relevant Experience", sorted(df["relevant_experience"].dropna().unique())
                )
                career_switch = c.selectbox(
                    "Career Switch Willingness",
                    sorted(df["career_switch_willingness"].dropna().unique())
                )
                previous_ctc = a.number_input("Previous CTC (LPA)", min_value=0.0, value=5.0)

            with t4:
                st.markdown("### Job & Company Details")
                a, b, c = st.columns(3)
                expected_ctc = a.number_input("Expected CTC (LPA)", min_value=0.0, value=7.0)
                company_tier = b.selectbox(
                    "Company Tier", sorted(df["company_tier"].dropna().unique())
                )
                job_match = c.selectbox(
                    "Job Role Match", sorted(df["job_role_match"].dropna().unique())
                )
                a, b, c = st.columns(3)
                competition = a.selectbox(
                    "Competition Level", sorted(df["competition_level"].dropna().unique())
                )
                bond = b.selectbox(
                    "Bond Requirement", sorted(df["bond_requirement"].dropna().unique())
                )

            with t5:
                st.markdown("### Other Preferences")
                a, b, c = st.columns(3)
                notice = a.number_input("Notice Period (Days)", min_value=0, max_value=365, value=0)
                layoff = b.selectbox(
                    "Layoff History", sorted(df["layoff_history"].dropna().unique())
                )
                employment_gap = c.number_input(
                    "Employment Gap (Months)", min_value=0, max_value=120, value=0
                )
                relocation = a.selectbox(
                    "Relocation Willingness", sorted(df["relocation_willingness"].dropna().unique())
                )

        submit = st.form_submit_button(
            "Predict",
        )

    if submit:

        answers = {
            "age_years": age,
            "gender": gender,
            "degree_specialization": degree,
            "ssc_percentage": ssc,
            "hsc_percentage": hsc,
            "degree_percentage": degree_percentage,
            "technical_score": technical,
            "aptitude_score": aptitude,
            "communication_score": communication,
            "skills_match_percentage": skills,
            "certifications_count": certifications,
            "internship_experience": internship,
            "years_of_experience": experience,
            "relevant_experience": relevant,
            "career_switch_willingness": career_switch,
            "previous_ctc_lpa": previous_ctc,
            "expected_ctc_lpa": expected_ctc,
            "company_tier": company_tier,
            "job_role_match": job_match,
            "competition_level": competition,
            "bond_requirement": bond,
            "notice_period_days": notice,
            "layoff_history": layoff,
            "employment_gap_months": employment_gap,
            "relocation_willingness": relocation
        }

        input_df = pd.DataFrame([answers])

        probability = model.predict_proba(input_df[features])[0][1]

        accepted = probability >= threshold

        st.divider()

        st.subheader("Prediction Result")

        left, right = st.columns([1, 1])

        with left:
            result_image = "placed_insight.png" if accepted else "not_placed_insight.png"
            if (IMAGE_DIR / result_image).exists():
                st.image(str(IMAGE_DIR / result_image), use_container_width=True)

        with right:
            st.markdown("**Acceptance Meter**")
            st.progress(float(min(max(probability, 0.0), 1.0)))
            gap = (probability - threshold) * 100
            with st.container(key="result_metric"):
                st.metric("Acceptance Probability", f"{probability * 100:.2f}%")
            with st.expander("Data used for this prediction"):
                st.dataframe(
                    input_df.T.astype(str)
                        .rename(columns={0: "Value"})
                        .rename(index=lambda s: s.replace("_", " ").title()),
                    use_container_width=True,
                    height=490,
                )
            
def main():
    st.set_page_config(page_title="Job Acceptance Prediction System", layout="wide")
    primary = st.get_option("theme.primaryColor") or "#4F46E5"
    st.markdown(
        f"<style>:root {{--primary: {primary};}}\n{STYLE_PATH.read_text()}</style>",
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="hero"><h1>Job Acceptance Prediction System</h1></div>',
        unsafe_allow_html=True
    )

    bundle = joblib.load(MODEL_PATH)
    model = bundle["pipeline"]
    threshold = bundle["threshold"]
    features = model.named_steps["preprocessor"].feature_names_in_

    dashboard, predict = st.tabs(["Dashboard", "Predict Candidate"])

    with dashboard:
        show_dashboard(model, threshold, features)
    with predict:
        show_prediction(model, threshold, features)

if __name__ == "__main__":
    main()