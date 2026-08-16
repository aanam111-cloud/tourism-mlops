"""
Streamlit App – Wellness Tourism Package Predictor
Loads the model from tourism_project/deployment/ and predicts purchase likelihood.
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Wellness Tourism Predictor | Visit with Us",
    page_icon="✈️",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "best_model.joblib"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Run the training pipeline first.")
        return None
    return joblib.load(MODEL_PATH)


def main():
    # --- HERO HEADER SECTION ---
    st.markdown(
        """
        <div style="background-color: #f0f7f6; padding: 2.5rem; border-radius: 12px; margin-bottom: 2rem; border-left: 8px solid #0d6efd;">
            <h1 style="color: #1e293b; margin-top: 0;">✈️ Wellness Tourism Package Predictor</h1>
            <p style="color: #475569; font-size: 1.15rem; margin-bottom: 0;">
                Optimize your conversions. Predict whether an inquiring tourist is highly likely to purchase the
                <strong>Premium Wellness Package</strong> before finalizing your sales team outreach strategy.
            </p>
        </div>
        """, unsafe_allow_html=True,
    )

    model = load_model()
    if model is None:
        st.stop()

    # --- INPUT STRUCTURED TABS ---
    st.subheader("📊 Customer Profile Evaluation")
    tab1, tab2, tab3 = st.tabs([
        "👤 Personal & Demographic Info",
        "🧳 Trip & Accommodation Preferences",
        "📞 Sales Interaction Metrics"
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=80, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            maritalstatus = st.selectbox(
                "Marital Status", ["Single", "Married", "Divorced", "Unmarried"]
            )
        with col2:
            occupation = st.selectbox(
                "Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]
            )
            designation = st.selectbox(
                "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
            )
            monthlyincome = st.number_input(
                "Monthly Income", min_value=1000, max_value=100000, value=22000, step=500
            )

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            numberofpersonvisiting = st.number_input(
                "Number of Persons Visiting", min_value=1, max_value=5, value=2
            )
            numberofchildrenvisiting = st.number_input(
                "Number of Children Visiting", min_value=0, max_value=3, value=0
            )
            numberoftrips = st.number_input(
                "Number of Trips (annual avg)", min_value=0, max_value=25, value=2
            )
        with col2:
            productpitched = st.selectbox(
                "Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
            )
            preferredpropertystar = st.selectbox(
                "Preferred Property Star", [3.0, 4.0, 5.0], index=0
            )
            owncar = st.selectbox(
                "Owns Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
            )

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
            citytier = st.selectbox("City Tier", [1, 2, 3], index=0)
            passport = st.selectbox(
                "Holds Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
            )
        with col2:
            numberoffollowups = st.number_input(
                "Number of Follow-ups", min_value=0, max_value=10, value=3
            )
            durationofpitch = st.number_input(
                "Duration of Pitch (minutes)", min_value=1, max_value=40, value=10
            )
            pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", 1, 5, 3)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- PREDICTION TRIGGER ---
    if st.button("🔍 Run ML Conversion Forecast", type="primary", use_container_width=True):
        input_data = {
            "Age": age,
            "TypeofContact": typeofcontact,
            "CityTier": citytier,
            "DurationOfPitch": durationofpitch,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": numberofpersonvisiting,
            "NumberOfFollowups": numberoffollowups,
            "ProductPitched": productpitched,
            "PreferredPropertyStar": preferredpropertystar,
            "MaritalStatus": maritalstatus,
            "NumberOfTrips": numberoftrips,
            "Passport": passport,
            "PitchSatisfactionScore": pitchsatisfactionscore,
            "OwnCar": owncar,
            "NumberOfChildrenVisiting": numberofchildrenvisiting,
            "Designation": designation,
            "MonthlyIncome": monthlyincome,
        }
        input_df = pd.DataFrame([input_data])

        try:
            proba = model.predict_proba(input_df)[0, 1]
            pred = model.predict(input_df)[0]

            # --- OUTPUT SCORECARD PRESENTATION ---
            st.markdown("""<hr style="border: 1px solid #cbd5e1;">""", unsafe_allow_html=True)
            st.markdown("### 🎯 Predictive Assessment Results")

            with st.container(border=True):
                m1, m2, m3 = st.columns(3)

                if pred == 1:
                    m1.metric("Conversion Result", "✅ Will Purchase")
                else:
                    m1.metric("Conversion Result", "❌ Will Not Purchase")

                m2.metric("Purchase Likelihood", f"{proba:.1%}")

                confidence_str = (
                    "🔥 High" if abs(proba - 0.5) > 0.3
                    else "⚡ Medium" if abs(proba - 0.5) > 0.15
                    else "⚠️ Low"
                )
                m3.metric("Model Confidence Level", confidence_str)

                st.markdown("<br>", unsafe_allow_html=True)
                st.progress(float(proba), text=f"Probability Distribution: {proba:.1%}")

            st.markdown("<br>", unsafe_allow_html=True)

            if pred == 1:
                st.success(
                    "💡 **Strategic Advisory:** Strong high-intent prospect identified! "
                    "Prioritize immediate personalized luxury sales outreach to seal this package booking."
                )
            else:
                st.info(
                    "💡 **Strategic Advisory:** Lower immediate conversion likelihood. "
                    "Incorporate customer into long-term nurturing workflows or standard newsletter promos."
                )

            with st.expander("🔬 Review raw structural features passed to model pipeline"):
                st.dataframe(input_df.T.rename(columns={0: "Parsed Feature Values"}), use_container_width=True)

        except Exception as e:
            st.error(f"Prediction Pipeline Interrupted: {e}")


if __name__ == "__main__":
    main()
