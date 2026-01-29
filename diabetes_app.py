import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Load model
loaded_model = pickle.load(open("project_model.sav", "rb"))

def diabetes_prediction(input_data):
    input_array = np.asarray(input_data, dtype=float).reshape(1, -1)
    prediction = loaded_model.predict(input_array)

    if hasattr(loaded_model, "predict_proba"):
        proba = loaded_model.predict_proba(input_array)[0][1]
    else:
        proba = 1.0 if prediction[0] == 1 else 0.0

    return prediction[0], proba


def main():
    st.set_page_config(
        page_title="Diabetes Predictor Dashboard",
        page_icon="💉",
        layout="wide"
    )

    # Custom CSS
    st.markdown(
        """
        <style>
        .stApp { background-color: #f0f8ff; }

        h1, h2, h3 {
            color: #0f172a !important;
            text-align: center;
        }

        label {
            color: #0f172a !important;
            font-weight: 600;
        }

        input {
            color: #0f172a !important;
            background-color: white !important;
        }

        button {
            background-color: #1f77b4 !important;
            color: white !important;
            border-radius: 10px;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title
    st.title("💉 Diabetes Health Indicator Dashboard")
    st.markdown("Fill in your health data to see your diabetes risk probability.")

    # Sidebar
    st.sidebar.image("logo.png", width=200)
    st.sidebar.title("About")
    st.sidebar.info(
        "This dashboard predicts diabetes risk using health indicators.\n"
        "Powered by a Machine Learning model."
    )

    # Inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        HighBP = st.number_input("HighBP (0 or 1)", 0, 1)
        Stroke = st.number_input("Stroke (0 or 1)", 0, 1)
        HvyAlcoholConsump = st.number_input("HvyAlcoholConsump (0 or 1)", 0, 1)
        DiffWalk = st.number_input("DiffWalk (0 or 1)", 0, 1)

    with col2:
        HighChol = st.number_input("HighChol (0 or 1)", 0, 1)
        HeartDiseaseorAttack = st.number_input("HeartDiseaseorAttack (0 or 1)", 0, 1)
        AnyHealthcare = st.number_input("AnyHealthcare (0 or 1)", 0, 1)
        PhysActivity = st.number_input("PhysActivity (0 or 1)", 0, 1)

    with col3:
        BMI = st.slider("BMI", min_value=1.0, max_value=27.0, value=22.0, step=0.1)
        #BMI = st.number_input("BMI", min_value=0.0, format="%.1f")
        GenHlth = st.number_input("GenHlth (1–5)", 1, 5)
        Sex = st.number_input("Sex (0 = Female, 1 = Male)", 0, 1)
        Age = st.number_input("Age (1–13)", 1, 13)

    # Prediction
    if st.button("🔎 Check Diabetes Risk"):
        pred, risk_proba = diabetes_prediction([
            HighBP, HighChol, BMI, Stroke, HeartDiseaseorAttack,
            PhysActivity, HvyAlcoholConsump, AnyHealthcare,
            GenHlth, DiffWalk, Sex, Age
        ])

        st.markdown("---")
        st.markdown(
            "<h2 style='text-align:center;'>Prediction Result</h2>",
            unsafe_allow_html=True
        )

        if pred == 1:
            st.markdown(
                """
                <div style="background:#fdecea; border-left:8px solid #e53935;
                padding:20px; border-radius:12px; color:#7f1d1d;
                font-size:18px; text-align:center; font-weight:600;">
                ⚠️ High Risk: You may have diabetes.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="background:#e8f5e9; border-left:8px solid #2e7d32;
                padding:20px; border-radius:12px; color:#14532d;
                font-size:18px; text-align:center; font-weight:600;">
                ✅ Low Risk: You are likely Non-Diabetic.
                </div>
                """,
                unsafe_allow_html=True
            )

        # Risk bar
        st.markdown("### Diabetes Risk Probability")
        st.progress(int(risk_proba * 100))

        # Metrics
        st.markdown("### Key Health Metrics")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("BMI", BMI)
        col_b.metric("Age", Age)
        col_c.metric("General Health", GenHlth)

        # Chart
        st.markdown("### Health Metrics Overview")
        fig, ax = plt.subplots()
        ax.bar(["BMI", "GenHlth", "Age"], [BMI, GenHlth, Age])
        ax.set_ylim(0, max(30, BMI + 5))
        st.pyplot(fig)


if __name__ == "__main__":
    main()
