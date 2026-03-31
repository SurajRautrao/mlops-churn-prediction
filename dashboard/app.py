import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from explain import explain_prediction
import os
from catboost import Pool

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Churn Intelligence Dashboard", layout="wide")

st.markdown("""
# 📊 Churn Intelligence Dashboard
### Customer Risk Profiling & Retention Insights
""")

# ===================== SIDEBAR INPUT =====================
st.sidebar.header("🧾 Customer Configuration")

# ===================== CUSTOMER INFO =====================
with st.sidebar.expander("👤 Customer Information", expanded=True):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

# ===================== SERVICES =====================
with st.sidebar.expander("📞 Services"):
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# ===================== ADD-ONS =====================
with st.sidebar.expander("🔐 Add-on Services"):
    online_sec = st.selectbox("Online Security", ["Yes", "No"])
    backup = st.selectbox("Online Backup", ["Yes", "No"])
    device = st.selectbox("Device Protection", ["Yes", "No"])
    tech = st.selectbox("Tech Support", ["Yes", "No"])

# ===================== ENTERTAINMENT =====================
with st.sidebar.expander("🎬 Entertainment"):
    tv = st.selectbox("Streaming TV", ["Yes", "No"])
    movies = st.selectbox("Streaming Movies", ["Yes", "No"])

# ===================== BILLING =====================
with st.sidebar.expander("💳 Billing"):
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
    monthly = st.number_input("Monthly Charges", value=70.0)
    total = st.number_input("Total Charges", value=800.0)

# ===================== USER DATA =====================
user_data = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple,
    "InternetService": internet,
    "OnlineSecurity": online_sec,
    "OnlineBackup": backup,
    "DeviceProtection": device,
    "TechSupport": tech,
    "StreamingTV": tv,
    "StreamingMovies": movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total
}

# ===================== MAIN TABS =====================
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "🧠 Explainability", "💡 Business Insights"])

# ===================== TAB 1: PREDICTION =====================
with tab1:
    st.header("🔮 Churn Prediction")

    if st.button("Predict Churn"):

        response = requests.post("http://127.0.0.1:8000/predict", json=user_data)

        if response.status_code == 200:
            result = response.json()

            if "churn_probability" in result:
                st.session_state["prob"] = result["churn_probability"]
                prob = st.session_state["prob"]

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Churn Probability", f"{prob:.2f}")

                col1.metric("Churn Probability", f"{prob:.2%}")
                col2.metric("Risk Level", 
                            "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low")

                col3.metric("Monthly Charges", f"€{monthly:.2f}")
                col4.metric("Tenure", f"{tenure} months")

                st.progress(int(prob * 100))
                if prob > 0.7:
                    st.error("Customer is highly likely to churn. Immediate intervention recommended.")
                elif prob > 0.4:
                    st.warning("Customer shows moderate churn risk. Monitor engagement.")
                else:
                    st.success("Customer is stable with low churn risk.")
            else:
                st.error(f"API Error: {result}")
        else:
            st.error(f"API Error: {response.text}")

# ===================== TAB 2: SHAP =====================
import shap
from catboost import Pool
model = joblib.load("models/catboost_model.pkl")

with tab2:
    prob = st.session_state.get("prob", None)
    if prob is None:
        st.warning("⚠️ Please run prediction first to view insights")
    else:    
        st.header("🧠 Model Explainability (SHAP)")
        # ------------------ WATERFALL (Single Prediction) ------------------
        st.subheader("🔍 Why this customer might churn") 
        feature_names, shap_vals = explain_prediction(user_data)

        shap_df = pd.DataFrame({
            "feature": feature_names,
            "importance": shap_vals
        })

        shap_df["abs"] = shap_df["importance"].abs()
        top5 = shap_df.sort_values(by="abs", ascending=False).head(5)
        st.markdown("### 🧾 Summary")
        risk_driver = top5.iloc[0]["feature"]
        st.info(f"""
        The primary driver of churn for this customer is **{risk_driver}**.
        Addressing this factor can significantly reduce churn probability.
        """)
        col1, col2 = st.columns(2, gap="large")

        with col1:
            # Waterfall plot
            df_single = pd.DataFrame([user_data])

            cat_cols = df_single.select_dtypes(include=["object"]).columns.tolist()
            pool = Pool(df_single, cat_features=cat_cols)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(pool)

            # Handle output
            if isinstance(shap_values, list):
                vals = shap_values[1][0]
            else:
                vals = shap_values[0]

            # Create SHAP Explanation object
            explanation = shap.Explanation(
                values=vals,
                base_values=explainer.expected_value,
                data=df_single.iloc[0],
                feature_names=df_single.columns
            )

            fig = plt.figure()
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(fig)

        with col2:
            st.markdown("### 📌 Key Insights")

            insights = []

            for _, row in top5.iterrows():
                feature = row["feature"]
                value = row["importance"]

                feature_value = user_data.get(feature, "N/A")

                feature_value = str(feature_value).replace("_", " ")

                if value > 0:
                    insights.append(f"🔺 **{feature} = {feature_value}** is increasing churn risk")
                else:
                    insights.append(f"🔻 **{feature} = {feature_value}** is reducing churn risk")

            for i in insights:
                st.write(i)      

        col1, col2 = st.columns(2, gap="large")

        with col1:
            # ------------------ BEESWARM (Global Insight) ------------------
            st.subheader("🌍 Global Feature Impact (Beeswarm)")
            try:
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                data_path = os.path.join(BASE_DIR, "..", "data", "telco.csv")
                df_sample = pd.read_csv(data_path)

                df_sample["TotalCharges"] = pd.to_numeric(df_sample["TotalCharges"], errors="coerce")
                df_sample = df_sample.fillna(0)

                X_sample = df_sample.drop(["Churn", "customerID"], axis=1)
                X_sample = X_sample.sample(300, random_state=42)

                cat_cols = X_sample.select_dtypes(include=["object"]).columns.tolist()

                for col in cat_cols:
                    X_sample[col] = X_sample[col].astype(str)

                pool_sample = Pool(X_sample, cat_features=cat_cols)

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(pool_sample)

                fig2 = plt.figure()
                shap.summary_plot(shap_values, X_sample, show=False)

                st.pyplot(fig2)

            except Exception as e:
                st.error(f"Beeswarm error: {e}")
            st.caption("Each point represents a customer. Color indicates feature value (low → high).")

        with col2:
            # ------------------ DEPENDENCE PLOT ------------------
            st.subheader("📈 Feature Interaction (Dependence Plot)")

            import numpy as np
            import matplotlib.pyplot as plt

            try:
                important_features = ["Contract","gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService",
                                        "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
                                        "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
                                        "PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"]

                selected_feature = st.selectbox(
                    "Select feature for analysis",
                    important_features
                )

                # Handle SHAP values
                if isinstance(shap_values, list):
                    shap_vals = shap_values[1]
                else:
                    shap_vals = shap_values

                shap_vals = np.array(shap_vals)

                # Remove last column (CatBoost expected value)
                if shap_vals.shape[1] == X_sample.shape[1] + 1:
                    shap_vals = shap_vals[:, :-1]

                # KEY FIX: Let SHAP draw directly
                shap.dependence_plot(
                    selected_feature,
                    shap_vals,
                    X_sample,
                    interaction_index="auto",
                    show=False
                )

                # Get current figure instead of creating new one
                fig = plt.gcf()
                st.pyplot(fig)
                st.caption(f"""This plot shows how **{selected_feature}** impacts churn probability. Each dot represents a customer. Spread indicates variability in behavior.""")

                plt.clf()  # clear figure to avoid overlap

            except Exception as e:
                st.error(f"Dependence plot error: {e}")
        st.markdown("### 📊 Most Important Features (Global)")

        importance = np.abs(shap_vals).mean(axis=0)
        feature_importance = pd.DataFrame({
            "feature": X_sample.columns,
            "importance": importance
        }).sort_values(by="importance", ascending=False).head(5)

        st.bar_chart(feature_importance.set_index("feature"))

# ===================== TAB 3: BUSINESS INSIGHTS =====================
with tab3:
    st.header("💡 Business Insights & Recommendations")

    prob = st.session_state.get("prob", None)

    # 🚫 If prediction not done
    if prob is None:
        st.warning("⚠️ Please run prediction first to view insights")
    
    # ✅ Only show insights AFTER prediction
    else:
        st.subheader("📊 Key Drivers for This Customer")

        if contract == "Month-to-month":
            st.write("📄 Month-to-month contract increases churn risk")

        if monthly > 80:
            st.write("💰 High monthly charges are contributing to churn")

        if tech == "No":
            st.write("📞 Lack of tech support increases dissatisfaction")

        if tenure < 12:
            st.write("⏳ Low tenure indicates weak customer loyalty")

        st.subheader("🎯 Recommended Actions")

        # Risk-based recommendation
        if prob < 0.4:
            st.info("Customer is stable — focus on upselling premium services")

        elif prob < 0.7:
            st.warning("Moderate churn risk — consider engagement strategies")

        else:
            st.error("High churn risk — immediate action required")

        # Rule-based recommendations
        if contract == "Month-to-month":
            st.success("Offer long-term contract incentives (1-year or 2-year plans)")

        if monthly > 80:
            st.success("Provide discount or bundled service packages")

        if tech == "No":
            st.success("Promote tech support subscription")

        if tenure < 12:
            st.success("Engage customer with onboarding or loyalty programs")


