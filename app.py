import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import os
import random

# --- Page Configuration ---
st.set_page_config(
    page_title="FinanSage AI Portal",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Function to Load Custom CSS ---
@st.cache_data
def load_css(file_name):
    """Loads a CSS file from the same directory as the script."""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            st.markdown('<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Error: The 'style.css' file was not found. Please ensure it is in the main project directory.")

# --- Asset Caching ---
@st.cache_data
def load_data(path):
    # ... (code unchanged)
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Error: The data file was not found at '{path}'. Please ensure 'bank_data_final.csv' is in the 'data' folder.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

@st.cache_resource
def train_model(df):
    # ... (code unchanged)
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    df_copy = df.copy()
    df_copy['y'] = df_copy['y'].map({'yes': 1, 'no': 0})
    pii_columns = ['CustomerID', 'FirstName', 'LastName', 'MobileNumber', 'Email', 'Address', 'AccountNumber', 'IFSCCode', 'LoginUserID']
    X = df_copy.drop(columns=pii_columns + ['y'])
    y = df_copy['y']
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_features), ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])
    pipeline.fit(X, y)
    return pipeline, X.columns

# --- Employee Portal Pages ---
def page_analytics(df):
    st.header("📊 Customer Analytics Dashboard")
    st.subheader("Key Performance Indicators (KPIs)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{df.shape[0]:,}")
    subscription_rate = df['y'].value_counts(normalize=True).get('yes', 0) * 100
    col2.metric("Subscription Rate", f"{subscription_rate:.2f}%")
    avg_balance = df['balance'].mean()
    col3.metric("Avg. Balance (₹)", f"{avg_balance:,.0f}")
    avg_age = df['age'].mean()
    col4.metric("Avg. Customer Age", f"{avg_age:.1f}")
    st.markdown("---")
    st.subheader("Customer Demographics")
    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(px.histogram(df, x='age', nbins=40, title='Age Distribution'), use_container_width=True)
    with col2: st.plotly_chart(px.bar(df['job'].value_counts().reset_index(), x='job', y='count', title='Job Distribution'), use_container_width=True)

def page_employee_bots(df):
    st.header("🤖 AI Bot Console")
    st.markdown("Activate intelligent bots to automate and enhance your workflow.")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("🔔 At-Risk Customer Bot")
            st.write("This bot identifies customers whose balance has dropped significantly, flagging them as potential churn risks.")
            if st.button("Scan for At-Risk Customers"):
                at_risk_df = df[df['balance'] < 10000].head(3)
                st.error("High-Priority Alerts Found!")
                for _, row in at_risk_df.iterrows():
                    st.warning(f"**{row['FirstName']} {row['LastName']}** (Balance: ₹{row['balance']:,}) - Balance is critically low. Recommend outreach.", icon="🚨")
    with col2:
        with st.container(border=True):
            st.subheader("🎯 Daily Lead Bot")
            st.write("Generates a fresh, prioritized list of the top 5 customers to contact today for term deposit campaigns.")
            if st.button("Generate Today's Leads"):
                model = st.session_state.model
                model_columns = st.session_state.model_columns
                unsubscribed_df = df[df['y'] == 'no'].copy()
                leads_to_predict = unsubscribed_df[model_columns]
                predictions = model.predict_proba(leads_to_predict)[:, 1]
                unsubscribed_df['Subscription Likelihood'] = predictions
                top_leads = unsubscribed_df.sort_values(by='Subscription Likelihood', ascending=False).head(5)
                st.success("Today's Top 5 Leads Generated!")
                for _, row in top_leads.iterrows():
                    st.info(f"**{row['FirstName']} {row['LastName']}** (Phone: {row['MobileNumber']}) - Likelihood: **{row['Subscription Likelihood']:.1%}**", icon="📞")

def page_customer_360(df, model, model_columns):
    st.header("👤 Customer 360° View")
    df['DisplayName'] = df['FirstName'] + ' ' + df['LastName'] + ' (ID: ' + df['CustomerID'].astype(str) + ')'
    selected_customer_name = st.selectbox("Select Customer", df['DisplayName'])
    if selected_customer_name:
        customer_data = df[df['DisplayName'] == selected_customer_name].iloc[0]
        st.subheader(f"Profile: {customer_data['FirstName']} {customer_data['LastName']}")
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Mobile Number", customer_data['MobileNumber'], disabled=True)
            st.text_input("Email", customer_data['Email'], disabled=True)
        with col2:
            st.text_input("Job", customer_data['job'], disabled=True)
            st.text_input("Account Balance (₹)", f"{customer_data['balance']:,}", disabled=True)
        st.markdown("---")
        st.subheader("AI Propensity Score")
        customer_to_predict = customer_data[model_columns].to_frame().T
        prediction_proba = model.predict_proba(customer_to_predict)[0][1]
        col1, col2 = st.columns([1,2])
        with col1: st.metric("Term Deposit Subscription Likelihood", f"{prediction_proba:.1%}")
        with col2:
            st.progress(float(prediction_proba))
            if prediction_proba > 0.5: st.success("HIGH-potential lead. Recommend contacting soon.")
            else: st.warning("LOW-potential lead. Nurture with general offers.")

# --- Customer Portal Pages ---
def page_account_summary():
    customer_data = st.session_state.customer_data
    st.header(f"Welcome Back, {customer_data['FirstName']}!")
    if 'accounts' not in st.session_state:
        if customer_data['job'] == 'student': st.session_state.accounts = {"Savings": customer_data['balance']}
        else: st.session_state.accounts = {"Checking": customer_data['balance'] * 0.4, "Savings": customer_data['balance'] * 0.6}
    st.subheader("Your Account Details")
    col1, col2 = st.columns(2)
    with col1: st.text_input("Account Number", value=customer_data['AccountNumber'], disabled=True)
    with col2: st.text_input("IFSC Code", value=customer_data['IFSCCode'], disabled=True)
    st.subheader("Account Balances")
    cols = st.columns(len(st.session_state.accounts))
    for i, (acc_name, acc_balance) in enumerate(st.session_state.accounts.items()): cols[i].metric(acc_name, f"₹{acc_balance:,.2f}")
    if 'transactions' not in st.session_state:
        st.session_state.transactions = [
            {"Date": (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'), "Description": "Supermarket", "Amount (₹)": -5210.50, "Category": "Groceries"},
            {"Date": (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d'), "Description": "Salary Credit", "Amount (₹)": 75000.00, "Category": "Income"},
        ]
    st.markdown("---")
    st.subheader("Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("📲 Send Money via UPI"):
            with st.form("upi_form", clear_on_submit=True):
                recipient_upi_id = st.text_input("Recipient UPI ID", "merchant@okbank")
                amount = st.number_input("Amount (₹)", min_value=1.0, step=10.0)
                debit_account = st.selectbox("From Account", list(st.session_state.accounts.keys()), key="upi_debit")
                if st.form_submit_button("Send via UPI"):
                    if amount > st.session_state.accounts[debit_account]: st.error("Insufficient balance.")
                    else:
                        st.session_state.accounts[debit_account] -= amount
                        new_tx = {"Date": datetime.now().strftime('%Y-%m-%d'), "Description": f"UPI to {recipient_upi_id}", "Amount (₹)": -amount, "Category": "Payments"}
                        st.session_state.transactions.insert(0, new_tx)
                        st.toast(f"✅ ₹{amount} sent successfully!", icon="🎉"); st.rerun()
    with col2:
        with st.expander("🏦 Within-Bank Transfer"):
            with st.form("transfer_form", clear_on_submit=True):
                recipient_list = st.session_state.all_customers[st.session_state.all_customers['CustomerID'] != customer_data['CustomerID']]
                recipient_name = st.selectbox("Select Recipient", recipient_list['FirstName'] + ' ' + recipient_list['LastName'])
                amount = st.number_input("Amount (₹)", min_value=1.0, step=100.0)
                debit_account = st.selectbox("From Account", list(st.session_state.accounts.keys()), key="transfer_debit")
                if st.form_submit_button("Transfer Money"):
                    if amount > st.session_state.accounts[debit_account]: st.error("Insufficient balance.")
                    else:
                        st.session_state.accounts[debit_account] -= amount
                        new_tx = {"Date": datetime.now().strftime('%Y-%m-%d'), "Description": f"Transfer to {recipient_name}", "Amount (₹)": -amount, "Category": "Transfers"}
                        st.session_state.transactions.insert(0, new_tx)
                        st.toast(f"✅ ₹{amount} transferred successfully!", icon="🎉"); st.rerun()
    st.markdown("---")
    st.subheader("Recent Transactions")
    st.dataframe(pd.DataFrame(st.session_state.transactions), use_container_width=True)

def page_algo_bots():
    st.header("🤖 Algo Savings & Investment Bots")
    st.markdown("Automate your finances with our smart bots. Activate them once and watch your wealth grow.")
    
    # ** FIX: Initialize all required bot states correctly **
    if 'bots' not in st.session_state:
        st.session_state.bots = {"round_up": False, "smart_transfer": False, "round_up_pot": 0.0}

    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("💰 Round-Up Savings Bot")
            st.write("Automatically rounds up your daily spends to the nearest ₹10 or ₹50 and invests the change into a liquid fund.")
            is_active = st.session_state.bots["round_up"]
            if is_active:
                if st.button("Deactivate Round-Up Bot"):
                    st.session_state.bots["round_up"] = False; st.toast("Round-Up Bot deactivated.", icon="⏸️"); st.rerun()
                if st.button("Simulate 1 Day of Spends"):
                    spends = [random.uniform(5, 500) for _ in range(5)]
                    rounded_up = sum([10 - (s % 10) for s in spends])
                    st.session_state.bots["round_up_pot"] += rounded_up
                    st.toast(f"₹{rounded_up:.2f} added to your pot!", icon="💰"); st.rerun()
            else:
                if st.button("Activate Round-Up Bot"):
                    st.session_state.bots["round_up"] = True; st.toast("Round-Up Bot activated!", icon="🚀"); st.rerun()
        with col2:
            st.metric("Your Round-Up Pot", f"₹{st.session_state.bots['round_up_pot']:.2f}")
            if st.session_state.bots["round_up"]: st.success("✅ ACTIVE")
            else: st.info("INACTIVE")
    
    with st.container(border=True):
        st.subheader("🎯 Goal-Based SIP Bot")
        st.write("Define your financial goals, and this bot will calculate the required SIP and help you start.")
        goal = st.text_input("What is your financial goal?", "iPhone 17 Pro")
        target_amount = st.number_input("Target Amount (₹)", min_value=10000, value=180000)
        target_year = st.slider("Target Year", datetime.now().year + 1, datetime.now().year + 10, datetime.now().year + 2)
        years_to_go = target_year - datetime.now().year
        monthly_sip = (target_amount * (0.12/12)) / (((1 + 0.12/12)**(years_to_go*12)) - 1) if years_to_go > 0 else target_amount / 12
        col1, col2 = st.columns([2,1])
        with col1:
            st.metric(f"Required Monthly SIP for '{goal}'", f"₹{monthly_sip:,.0f}")
            st.info("Suggestion: A Flexi Cap or Index Fund would be suitable for this goal horizon.", icon="💡")
        with col2:
            if st.button("🚀 Start this SIP Plan", use_container_width=True):
                st.success(f"Congratulations! Your SIP of ₹{monthly_sip:,.0f}/month for '{goal}' has been simulated.")
                st.balloons()

# --- Login & Portal Logic ---
def show_login_page(df):
    st.markdown("<h1 style='text-align: center;'>🔐 FinanSage AI Portal</h1>", unsafe_allow_html=True)
    login_tab, create_account_tab = st.tabs(["Login to Your Account", "Open a New Account"])
    with login_tab:
        col1, col2 = st.columns(2)
        with col1:
            with st.form("employee_login"):
                st.subheader("🏦 Bank Employee Login")
                emp_user = st.text_input("Username", value="admin")
                emp_pass = st.text_input("Password", type="password", value="password123")
                if st.form_submit_button("Login as Employee"):
                    employee_creds = {"admin": "password123"}
                    if emp_user in employee_creds and emp_pass == employee_creds[emp_user]:
                        st.session_state.logged_in = True; st.session_state.user_type = "Employee"; st.session_state.username = emp_user; st.toast(f"Welcome, {emp_user}!", icon="👋"); st.rerun()
                    else: st.error("Invalid username or password")
        with col2:
            with st.form("customer_login"):
                st.subheader("👤 Customer Access Portal")
                customer_creds = dict(zip(df['LoginUserID'], df['MobileNumber'].astype(str)))
                cust_user_id = st.text_input("Customer Login ID", value="PriyaS2345")
                cust_pass = st.text_input("Password (use Mobile Number)", type="password", value="+91 9820012345")
                if st.form_submit_button("Login as Customer"):
                    if cust_user_id in customer_creds and cust_pass == customer_creds[cust_user_id]:
                        st.session_state.logged_in = True; st.session_state.user_type = "Customer";
                        st.session_state.customer_data = df[df['LoginUserID'] == cust_user_id].iloc[0].to_dict()
                        st.session_state.username = st.session_state.customer_data['FirstName']
                        st.toast(f"Welcome, {st.session_state.username}!", icon="👋"); st.rerun()
                    else: st.error("Invalid Login ID or Password")
    with create_account_tab:
        st.subheader("✨ Let's Get You Started")
        with st.form("new_account_form"):
            new_fname = st.text_input("First Name")
            new_lname = st.text_input("Last Name")
            new_mobile = st.text_input("Mobile Number (+91)")
            new_email = st.text_input("Email Address")
            if st.form_submit_button("Create My Account"):
                if all([new_fname, new_lname, new_mobile, new_email]):
                    new_cust_id = df['CustomerID'].max() + 1
                    new_acc_num = df['AccountNumber'].max() + 1
                    new_login_id = f"{new_fname.capitalize()}{new_lname[0].upper()}{str(new_mobile)[-4:]}"
                    st.success("🎉 Account Created Successfully!")
                    st.markdown("Please use these credentials to log in on the previous tab:")
                    st.text_input("Your New Customer Login ID", value=new_login_id, disabled=True)
                    st.text_input("Your Password (your Mobile Number)", value=new_mobile, disabled=True)
                    st.info("Note: This new account is for simulation only and will not be saved permanently.")
                else: st.error("Please fill in all the details.")

def show_employee_portal(df, model, model_columns):
    st.title(f"🏢 Employee Portal")
    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state.username.capitalize()}!")
        st.markdown("---")
        st.subheader("Your Performance")
        st.metric("Subscriptions Secured (Month)", "22"); st.metric("Conversion Rate", "18.5%"); st.progress(0.73, text="Monthly Target (73%)")
        st.markdown("---")
        selection = st.radio("Go to", ["📈 Customer Analytics", "👤 Customer 360° View", "🎯 AI Lead Finder", "🤖 AI Bot Console", "✨ Festive Offers"])
        st.markdown("---")
        if st.button("Logout"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()
    if selection == "📈 Customer Analytics": page_analytics(df)
    elif selection == "👤 Customer 360° View": page_customer_360(df, model, model_columns)
    elif selection == "🎯 AI Lead Finder": page_lead_finder(df, model, model_columns)
    elif selection == "🤖 AI Bot Console": page_employee_bots(df)
    elif selection == "✨ Festive Offers": page_bank_offers()

def show_customer_portal():
    st.title(f"👤 Customer Portal")
    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state.username}!")
        st.markdown("---")
        selection = st.radio("Go to", ["🏠 Account Summary", "🤖 Algo Savings", "💳 Cards & Loans", "💹 Investment Hub", "🧮 Financial Calculators", "❤️ Financial Health Check"])
        st.markdown("---")
        if st.button("Logout"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()
    if selection == "🏠 Account Summary": page_account_summary()
    elif selection == "🤖 Algo Savings": page_algo_bots()
    elif selection == "💳 Cards & Loans": page_cards_and_loans()
    elif selection == "💹 Investment Hub": page_investments()
    elif selection == "🧮 Financial Calculators": page_calculators()
    elif selection == "❤️ Financial Health Check": page_health_check()

# --- Main App ---
def main():
    load_css("style.css")
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    
    DATA_PATH = "data/bank_data_final.csv"
    df = load_data(DATA_PATH)
    
    theme_class = "dark-mode" if st.session_state.get('theme', 'light') == 'dark' else 'light-mode'
    st.markdown(f'<div class="main-container {theme_class}">', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("---")
        if st.toggle('🌙 Dark Mode', value=(st.session_state.get('theme', 'light') == 'dark')):
            if st.session_state.get('theme') != 'dark': st.session_state.theme = 'dark'; st.rerun()
        else:
            if st.session_state.get('theme') != 'light': st.session_state.theme = 'light'; st.rerun()

    if df is not None:
        if st.session_state.logged_in:
            if st.session_state.user_type == "Employee":
                model_pipeline, model_columns = train_model(df)
                st.session_state.model = model_pipeline
                st.session_state.model_columns = model_columns
                show_employee_portal(df, model_pipeline, model_columns)
            else: # Customer
                st.session_state.all_customers = df
                show_customer_portal()
        else:
            show_login_page(df)
            
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
