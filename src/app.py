import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 💡 App Config
st.set_page_config(page_title="CardDefender 🛡️", layout="centered")
st.title("💳 CardDefender - Credit Card Fraud Detector")

# 📁 Upload CSV
uploaded_file = st.file_uploader("📂 Upload your credit card transactions CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File Loaded Successfully!")
    
    # Show preview
    st.subheader("📄 Sample Data")
    st.write(df.head())

    # Drop 'Time' column if exists
    if 'Time' in df.columns:
        df.drop(['Time'], axis=1, inplace=True)

    # Prepare features
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 🕵️‍♂️ Isolation Forest
    model = IsolationForest(contamination=0.0017)
    model.fit(X_scaled)
    preds = model.predict(X_scaled)

    # Convert prediction (-1 = fraud → 1), (1 = legit → 0)
    preds = [1 if p == -1 else 0 for p in preds]
    df['Prediction'] = preds

    # 🔎 Show fraud results
    frauds = df[df['Prediction'] == 1]
    st.subheader("🕵️ Detected Fraudulent Transactions")
    st.write(f"🚨 Total Fraud Detected: **{len(frauds)}** out of {len(df)} transactions")
    st.dataframe(frauds)

    # 📊 Custom Bar Chart
    st.subheader("📊 Transaction Classification Overview")

    plot_data = df['Prediction'].value_counts().rename(index={0: 'Legit', 1: 'Fraud'}).reset_index()
    plot_data.columns = ['Transaction Type', 'Count']
    colors = ['#00b894', '#d63031']  # green for legit, red for fraud

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=plot_data, x='Transaction Type', y='Count', palette=colors, ax=ax)

    for i, row in plot_data.iterrows():
        ax.text(i, row['Count'] + 1000, f"{row['Count']:,}", ha='center', fontweight='bold')

    ax.set_title("Transaction Classification", fontsize=16, fontweight='bold')
    ax.set_ylabel("Number of Transactions")
    ax.set_xlabel("Type")
    sns.despine()

    st.pyplot(fig)

    # 📥 Download button
    st.download_button(
        label="💾 Download Fraud Results as CSV",
        data=frauds.to_csv(index=False),
        file_name='fraud_results.csv',
        mime='text/csv'
    )
