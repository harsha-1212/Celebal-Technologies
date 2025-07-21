import streamlit as st
import pandas as pd
import numpy as np
import joblib, shap, matplotlib.pyplot as plt, seaborn as sns

st.set_page_config(page_title="Iris Species Classifier",
                   layout="centered")


@st.cache_resource
def load_pipeline():
    model = joblib.load("iris_model.joblib")
    feature_names = joblib.load("iris_feature_names.joblib")
    
    target_names = joblib.load("iris_target_names.joblib")
    explainer = shap.TreeExplainer(model)
    return model, feature_names, target_names, explainer

model, feature_names, target_names, explainer = load_pipeline()


st.sidebar.header("Input Flower Measurements")

def user_input_widget():
    data = {}
    
    # feature ranges based on iris dataset statistics
    feature_ranges = {
        'sepal length (cm)': (4.3, 7.9, 5.8),
        'sepal width (cm)': (2.0, 4.4, 3.1), 
        'petal length (cm)': (1.0, 6.9, 3.8),
        'petal width (cm)': (0.1, 2.5, 1.2)
    }
    
    for feature in feature_names:
        min_val, max_val, default_val = feature_ranges[feature]
        data[feature] = st.sidebar.number_input(
            feature,
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=0.1,
            format="%.1f"
        )
    return pd.DataFrame([data])

input_df = user_input_widget()

# Main panel
st.title("🌸 Iris Species Classification")
st.markdown("Enter flower measurements on the left, then see real-time prediction and interpretation below.")

# Display input values
st.subheader("📊 Current Input Values")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Sepal Length", f"{input_df.iloc[0, 0]:.1f} cm")
with col2:
    st.metric("Sepal Width", f"{input_df.iloc[0, 1]:.1f} cm") 
with col3:
    st.metric("Petal Length", f"{input_df.iloc[0, 2]:.1f} cm")
with col4:
    st.metric("Petal Width", f"{input_df.iloc[0, 3]:.1f} cm")

# prediction
prediction = model.predict(input_df)[0]
probabilities = model.predict_proba(input_df)[0]
predicted_species = target_names[prediction]
confidence = probabilities[prediction]

st.subheader("🔮 Prediction Results")
col1, col2 = st.columns(2)

with col1:
    st.write(f"**Predicted Species: {predicted_species.title()}**")
    st.write(f"**Confidence: {confidence:.2%}**")
    
with col2:
    # Show probabilities for all classes
    prob_df = pd.DataFrame({
        'Species': target_names,
        'Probability': probabilities
    }).sort_values('Probability', ascending=False)
    
    st.write("**All Probabilities:**")
    for _, row in prob_df.iterrows():
        st.write(f"• {row['Species'].title()}: {row['Probability']:.3f}")

# Feature importance chart (global)
st.subheader("📈 Global Feature Importance")
imp_df = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)

fig1, ax1 = plt.subplots(figsize=(10, 6))
bars = ax1.barh(imp_df.index, imp_df.values, color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
ax1.set_xlabel("Feature Importance")
ax1.set_title("Random Forest Feature Importance for Iris Classification")
ax1.grid(axis='x', alpha=0.3)


for i, (idx, val) in enumerate(imp_df.items()):
    ax1.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
st.pyplot(fig1)

# SHAP waterfall for individual prediction
st.subheader("🔍 Individual Prediction Explanation (SHAP)")
st.write("This shows how each feature contributes to the prediction for your specific input:")

with st.spinner("Calculating SHAP values…"):
    shap_values = explainer(input_df.values)
    
    # For multi-class, show SHAP for the predicted class
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0, :, prediction], show=False)
    st.pyplot(fig2, use_container_width=True)

st.subheader("💡 Model Insights")
col1, col2 = st.columns(2)

with col1:
    st.info("""
    **Iris Species Characteristics:**
    
    🌺 **Setosa**: Generally smaller flowers, shorter petals
    
    🌸 **Versicolor**: Medium-sized flowers, moderate measurements
    
    🌷 **Virginica**: Larger flowers, longer petals and sepals
    """)

with col2:
    st.info(f"""
    **About this prediction:**
    
    📊 **Most Important Feature**: {imp_df.index[-1]}
    
    🎯 **Confidence Level**: {"High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"}
    
    🧠 **Model Type**: Random Forest (100 trees)
    """)

st.caption("🌱 Iris species classification demo - update any measurement to refresh results automatically.")
