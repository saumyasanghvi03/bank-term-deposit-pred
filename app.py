import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.cluster import KMeans

# --- Page Configuration ---
st.set_page_config(
    page_title="FinanSage AI Portal",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Asset Caching ---
@st.cache_data
def load_data(path):
    """Loads the dataset from a specified path with caching."""
    if not os.path.exists(path):
        st.error(f"Error: Dataset file not found at '{path}'. Please check the path.")
        return None
    return pd.read_csv(path)

@st.cache_resource
def train_model(df):
    """Trains the model and returns the pipeline with caching."""
    df_copy = df.copy()
    df_copy['y'] = df_copy['y'].map({'yes': 1, 'no': 0})
    X = df_copy.drop('y', axis=1)
    y = df_copy['y']
    
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])
    pipeline.fit(X, y)
    return pipeline

# --- Page Implementations for Employee Portal ---
def page_analytics(df):
    st.header("📊 Customer Analytics Dashboard")
    st.markdown("An in-depth look into the bank's customer base and marketing campaign performance.")
    
    tab1, tab2, tab3 = st.tabs(["📈 KPIs & Overview", "👥 Customer Segmentation", " campaing analysis"])
    
    with tab1:
        st.subheader("Key Performance Indicators (KPIs)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", f"{df.shape[0]:,}")
        subscription_rate = df['y'].value_counts(normalize=True)['yes'] * 100
        col2.metric("Subscription Rate", f"{subscription_rate:.2f}%")
        avg_balance = df['balance'].mean()
        col3.metric("Avg. Balance (€)", f"{avg_balance:,.0f}")
        avg_age = df['age'].mean()
        col4.metric("Avg. Customer Age", f"{avg_age:.1f}")

        st.markdown("---")
        st.subheader("Customer Demographics")
        col1, col2 = st.columns(2)
        with col1:
            fig_age = px.histogram(df, x='age', nbins=40, title='Age Distribution', template='plotly_white')
            st.plotly_chart(fig_age, use_container_width=True)
        with col2:
            fig_job = px.bar(df['job'].value_counts().reset_index(), x='job', y='count', title='Job Distribution', template='plotly_white')
            st.plotly_chart(fig_job, use_container_width=True)

    with tab2:
        st.subheader("Customer Segmentation with K-Means Clustering")
        st.markdown("Segmenting customers based on **Age** and **Balance** to identify distinct groups.")
        kmeans_df = df[['age', 'balance']].copy()
        scaler = StandardScaler()
        kmeans_df_scaled = scaler.fit_transform(kmeans_df)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(kmeans_df_scaled)
        kmeans_df['Cluster'] = kmeans.labels_.astype(str)
        fig_cluster = px.scatter(kmeans_df, x='age', y='balance', color='Cluster', title='Customer Segments (Age vs. Balance)', labels={'balance': 'Account Balance (€)'}, hover_data={'age': True, 'balance': ':.2f'})
        fig_cluster.update_layout(legend_title_text='Customer Segment')
        st.plotly_chart(fig_cluster, use_container_width=True)

    with tab3:
        st.subheader("Marketing Campaign Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig_month = px.bar(df.groupby(['month', 'y']).size().reset_index(name='count'), x='month', y='count', color='y', title='Subscriptions by Month', category_orders={"month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]})
            st.plotly_chart(fig_month, use_container_width=True)
        with col2:
            fig_poutcome = px.pie(df, names='poutcome', title='Previous Campaign Outcome', hole=0.3)
            st.plotly_chart(fig_poutcome, use_container_width=True)

def page_prediction(df, model_pipeline):
    st.header("🔮 Subscription Propensity AI")
    st.markdown("Predict a customer's likelihood to subscribe to a term deposit using our advanced AI model.")
    st.info("💡 **How it works:** Fill in the customer's details below. The model will analyze the information and provide a probability score for subscription.", icon="🤖")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Personal Details")
            age = st.number_input("Age", 18, 100, 40)
            job = st.selectbox("Job", df['job'].unique(), index=4)
            marital = st.selectbox("Marital Status", df['marital'].unique(), index=1)
            education = st.selectbox("Education", df['education'].unique(), index=1)
        with col2:
            st.subheader("Financial Status")
            default = st.selectbox("Has Credit in Default?", ["no", "yes"])
            balance = st.number_input("Avg. Yearly Balance (€)", -5000, 100000, 1500)
            housing = st.selectbox("Has Housing Loan?", ["no", "yes"])
            loan = st.selectbox("Has Personal Loan?", ["no", "yes"])
        with col3:
            st.subheader("Campaign History")
            contact = st.selectbox("Contact Type", df['contact'].unique(), index=2)
            day = st.slider("Last Contact Day", 1, 31, 15)
            month = st.selectbox("Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], index=4)
            duration = st.number_input("Last Contact Duration (sec)", 0, 5000, 300)
        
        submitted = st.form_submit_button("🧠 Predict Likelihood")

    if submitted:
        input_data = pd.DataFrame({'age': [age], 'job': [job], 'marital': [marital], 'education': [education], 'default': [default], 'balance': [balance], 'housing': [housing], 'loan': [loan], 'contact': [contact], 'day': [day], 'month': [month], 'duration': [duration], 'campaign': [1], 'pdays': [-1], 'previous': [0], 'poutcome': ['unknown']})
        prediction_proba = model_pipeline.predict_proba(input_data)[0][1]
        st.subheader("Prediction Result")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Subscription Probability", f"{prediction_proba:.1%}")
            if prediction_proba > 0.5: st.success("High Likelihood to Subscribe")
            else: st.error("Low Likelihood to Subscribe")
        with col2:
            st.progress(prediction_proba)
            st.markdown(f"There is a **{prediction_proba:.1%}** probability that this customer will subscribe. Consider prioritizing follow-up actions for customers with higher scores.")

# --- Page Implementations for Customer Portal ---
def page_account_summary():
    st.header(f"Welcome Back, {st.session_state.username.capitalize()}!")
    st.markdown("Here is a summary of your accounts and financial tools.")

    # Initialize account details in session state
    if 'accounts' not in st.session_state:
        st.session_state.accounts = {
            "Checking": 12540.50,
            "Savings": 7850.25
        }

    st.subheader("Account Balances")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Checking Account", f"€{st.session_state.accounts['Checking']:,.2f}")
    with col2:
        st.metric("Savings Account", f"€{st.session_state.accounts['Savings']:,.2f}")

    st.markdown("---")
    st.subheader("Quick Actions")

    with st.expander("💸 Make a Deposit"):
        account_to_deposit = st.selectbox("Select Account", list(st.session_state.accounts.keys()))
        deposit_amount = st.number_input("Deposit Amount (€)", 10.0, 10000.0, 50.0, 10.0, key="deposit")
        if st.button("Confirm Deposit"):
            st.session_state.accounts[account_to_deposit] += deposit_amount
            st.success(f"Deposit successful! New {account_to_deposit} balance: €{st.session_state.accounts[account_to_deposit]:,.2f}")
            st.rerun()

def page_calculator():
    st.header("🧮 Term Deposit Calculator")
    st.markdown("Calculate the future value of an investment with compound interest.")
    col1, col2, col3 = st.columns(3)
    with col1:
        principal = st.number_input("Principal Amount (€)", 100.0, 1000000.0, 5000.0, 100.0)
    with col2:
        rate = st.slider("Annual Interest Rate (%)", 0.1, 15.0, 5.5, 0.1)
    with col3:
        years = st.number_input("Investment Term (Years)", 1, 50, 10)
    
    if st.button("Calculate Future Value"):
        future_value = principal * ((1 + rate / 100) ** years)
        st.metric("Projected Value", f"€{future_value:,.2f}")
        growth_data = [{'Year': year, 'Value': principal * ((1 + rate / 100) ** year)} for year in range(years + 1)]
        growth_df = pd.DataFrame(growth_data)
        fig = px.area(growth_df, x='Year', y='Value', title='Investment Growth Over Time', markers=True)
        st.plotly_chart(fig, use_container_width=True)

# --- Login & Portal Logic ---
def show_login_page():
    st.markdown("<h1 style='text-align: center;'>🔐 FinanSage AI Portal</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Hardcoded credentials for simulation
    employee_creds = {"admin": "password123"}
    customer_creds = {"customer": "mumbai"}

    col1, col2 = st.columns(2)
    with col1:
        with st.form("employee_login"):
            st.subheader("🏦 Bank Employee Login")
            emp_user = st.text_input("Username", key="emp_user")
            emp_pass = st.text_input("Password", type="password", key="emp_pass")
            if st.form_submit_button("Login as Employee"):
                if emp_user in employee_creds and emp_pass == employee_creds[emp_user]:
                    st.session_state.logged_in = True
                    st.session_state.user_type = "Employee"
                    st.session_state.username = emp_user
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    with col2:
        with st.form("customer_login"):
            st.subheader("👤 Customer Access Portal")
            cust_user = st.text_input("Username", key="cust_user")
            cust_pass = st.text_input("Password", type="password", key="cust_pass")
            if st.form_submit_button("Login as Customer"):
                if cust_user in customer_creds and cust_pass == customer_creds[cust_user]:
                    st.session_state.logged_in = True
                    st.session_state.user_type = "Customer"
                    st.session_state.username = cust_user
                    st.rerun()
                else:
                    st.error("Invalid username or password")

def show_employee_portal(df, model):
    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state.username.capitalize()}!")
        st.markdown("---")
        page_options = {
            "📈 Customer Analytics": lambda: page_analytics(df),
            "🔮 Propensity AI": lambda: page_prediction(df, model)
        }
        selection = st.radio("Go to", list(page_options.keys()))
        st.markdown("---")
        if st.button("Logout"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
    
    st.title(f"🏢 Employee Portal: {selection}")
    page_options[selection]()

def show_customer_portal():
    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state.username.capitalize()}!")
        st.markdown("---")
        page_options = {
            "🏠 Account Summary": page_account_summary,
            "🧮 Financial Tools": page_calculator
        }
        selection = st.radio("Go to", list(page_options.keys()))
        st.markdown("---")
        if st.button("Logout"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
    
    st.title(f"👤 Customer Portal: {selection}")
    page_options[selection]()

# --- Main App ---
def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        if st.session_state.user_type == "Employee":
            df = load_data('data/bank-full.csv')
            if df is not None:
                model_pipeline = train_model(df)
                show_employee_portal(df, model_pipeline)
        else: # Customer
            show_customer_portal()
    else:
        show_login_page()

if __name__ == "__main__":
    main()
