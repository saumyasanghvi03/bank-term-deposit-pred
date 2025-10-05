# app.py - Stage 1

import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Bank Marketing Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Data Loading ---
@st.cache_data
def load_data(path):
    """Loads the dataset from a specified path."""
    if not os.path.exists(path):
        st.error(f"File not found at {path}. Please ensure the dataset is in the 'data' folder.")
        return None
    df = pd.read_csv(path)
    return df

# --- Main Application ---
def main():
    st.title("🏦 Bank Marketing Analytics Dashboard")
    st.markdown("An interactive dashboard for exploring customer data and predicting term deposit subscriptions.")

    # --- Load Data ---
    df = load_data('data/bank-full.csv')
    if df is None:
        return # Stop execution if data is not loaded

    # --- Sidebar ---
    st.sidebar.header("Navigation")
    # This is where we will add more pages later
    page = st.sidebar.radio("Go to", ["Exploratory Data Analysis (EDA)"])
    st.sidebar.markdown("---")
    st.sidebar.info("This project uses the UCI Bank Marketing dataset to predict customer behavior and provide financial tools.")


    # --- EDA Page ---
    if page == "Exploratory Data Analysis (EDA)":
        st.header("📊 Exploratory Data Analysis")
        st.markdown("Explore customer demographics, job types, and campaign outcomes.")

        # Display raw data
        if st.checkbox("Show raw data"):
            st.subheader("Raw Data")
            st.write(df)

        # Key Metrics
        st.subheader("Key Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", f"{df.shape[0]:,}")
        subscription_rate = (df['y'].value_counts(normalize=True)['yes'] * 100)
        col2.metric("Subscription Rate", f"{subscription_rate:.2f}%")
        avg_balance = df['balance'].mean()
        col3.metric("Average Balance (€)", f"{avg_balance:,.2f}")

        # --- Interactive Visualizations ---
        st.subheader("Interactive Charts")
        
        # Age Distribution
        fig_age = px.histogram(df, x='age', nbins=50, title='Customer Age Distribution',
                               labels={'age': 'Age'}, template='plotly_white')
        st.plotly_chart(fig_age, use_container_width=True)

        # Job Distribution vs Subscription
        fig_job = px.bar(df.groupby(['job', 'y']).size().reset_index(name='count'),
                         x='job', y='count', color='y',
                         title='Subscription Status by Job Type',
                         labels={'job': 'Job Type', 'count': 'Number of Customers', 'y': 'Subscribed'},
                         barmode='group', template='plotly_white')
        st.plotly_chart(fig_job, use_container_width=True)

        # Marital Status Distribution
        fig_marital = px.pie(df, names='marital', title='Customer Marital Status', hole=0.3)
        st.plotly_chart(fig_marital, use_container_width=True)


if __name__ == "__main__":
    main()
