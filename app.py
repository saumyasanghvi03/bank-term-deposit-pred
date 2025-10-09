import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="FinanSage AI Portal",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded" # Start expanded for better UX
)

# --- Function to Load Custom CSS (Robust Path) ---
@st.cache_data
def load_css(file_name):
    """Loads a CSS file from the same directory as the script."""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            st.markdown('<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Error: The 'style.css' file was not found. Please ensure it is in the main project directory.")

# --- Asset Caching (Robust Path) ---
@st.cache_data
def load_data(path):
    """Loads the dataset from a path relative to the script."""
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
    """Trains the model, EXCLUDING personal identifiable information (PII)."""
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

# --- All Page Functions (Defined Globally) ---

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

def page_prediction(df, model_pipeline, model_columns):
    st.header("🔮 Subscription Propensity AI")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Personal & Financial Details")
            age = st.number_input("Age", 18, 100, 40)
            job = st.selectbox("Job", df['job'].unique(), index=1)
            marital = st.selectbox("Marital Status", df['marital'].unique(), index=1)
            education = st.selectbox("Education", df['education'].unique(), index=1)
        with col2:
            st.subheader("Loan & Campaign Status")
            balance = st.number_input("Account Balance (₹)", -500000, 10000000, 50000)
            housing = st.selectbox("Has Housing Loan?", ["no", "yes"])
            loan = st.selectbox("Has Personal Loan?", ["no", "yes"])
            campaign = st.number_input("Number of Contacts in Campaign", 1, 100, 1)
        if st.form_submit_button("🧠 Predict Likelihood"):
            input_data_dict = {'age': [age], 'job': [job], 'marital': [marital], 'education': [education], 'balance': [balance], 'housing': [housing], 'loan': [loan], 'campaign': [campaign]}
            input_df = pd.DataFrame(input_data_dict)
            input_df_reordered = input_df.reindex(columns=model_columns, fill_value=0)
            prediction_proba = model_pipeline.predict_proba(input_df_reordered)[0][1]
            st.subheader("Prediction Result")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Subscription Probability", f"{prediction_proba:.1%}")
                if prediction_proba > 0.5: st.success("High Likelihood")
                else: st.error("Low Likelihood")
            with col2:
                st.progress(float(prediction_proba))
                st.markdown(f"There is a **{prediction_proba:.1%}** probability that this customer will subscribe.")

def page_bank_offers():
    st.header("✨ Festive Offers for Diwali 2025 ✨")
    offers = [
        {"title": "Dhanteras Gold Rush", "icon": "🪙", "rate": "Instant 5% Cashback", "benefit": "On Gold Jewellery & Coin Loans", "description": "Celebrate Dhanteras with a personal loan for gold purchases with zero processing fees and 5% cashback on the loan amount."},
        {"title": "Diwali Wheels of Joy", "icon": "🚗", "rate": "Starting at 8.25%", "benefit": "Zero Down Payment on Car Loans", "description": "Our special car loan offer comes with a rock-bottom interest rate and a zero down payment option for approved customers."},
        {"title": "Festive Home Makeover Loan", "icon": "🏡", "rate": "Attractive Low Interest", "benefit": "Quick Personal Loan for Renovations", "description": "Get a quick-disbursal personal loan up to ₹5 Lakhs for home improvements, painting, or buying new appliances."},
        {"title": "Diwali Dhamaka FD", "icon": "💰", "rate": "8.00% p.a.", "benefit": "Special High-Interest Fixed Deposit", "description": "A limited-period Fixed Deposit scheme offering a special high interest rate. Senior citizens get an additional 0.5%!"}
    ]
    for offer in offers:
        st.markdown(f"""
        <div class="offer-card">
            <h3>{offer['icon']} {offer['title']}</h3>
            <p><strong>Key Benefit:</strong> <span style="color: #E67E22; font-weight: bold;">{offer['benefit']}</span> | <strong>Offer Details:</strong> {offer['rate']}</p>
            <p>{offer['description']}</p>
        </div>
        """, unsafe_allow_html=True)

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

def page_lead_finder(df, model, model_columns):
    st.header("🎯 AI Lead Finder")
    st.markdown("A prioritized list of customers with the highest potential to subscribe to a term deposit.")
    unsubscribed_df = df[df['y'] == 'no'].copy()
    leads_to_predict = unsubscribed_df[model_columns]
    predictions = model.predict_proba(leads_to_predict)[:, 1]
    unsubscribed_df['Subscription Likelihood'] = predictions
    prioritized_leads = unsubscribed_df.sort_values(by='Subscription Likelihood', ascending=False)
    st.dataframe(prioritized_leads[['FirstName', 'LastName', 'MobileNumber', 'age', 'job', 'balance', 'Subscription Likelihood']],
                 use_container_width=True,
                 column_config={"Subscription Likelihood": st.column_config.ProgressColumn("Likelihood", format="%.2f", min_value=0, max_value=1)})

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
    # ... (Rest of the page logic is unchanged)
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

def page_cards_and_loans(): st.header("💳 Cards & Loans"); st.info("This feature is coming soon!")
def page_investments(): st.header("💹 Investment Hub"); st.info("This feature is coming soon!")
def page_calculators(): st.header("🧮 Financial Calculators"); st.info("This feature is coming soon!")

# --- Login & Portal Logic ---
def show_login_page(df):
    # ... (code for this page is unchanged)
    st.markdown("<h1 style='text-align: center;'>🔐 FinanSage AI Portal</h1>", unsafe_allow_html=True)
    st.markdown("---")
    customer_creds = dict(zip(df['LoginUserID'], df['MobileNumber'].astype(str)))
    employee_creds = {"admin": "password123"}
    col1, col2 = st.columns(2)
    with col1:
        with st.form("employee_login"):
            st.subheader("🏦 Bank Employee Login")
            emp_user = st.text_input("Username", value="admin")
            emp_pass = st.text_input("Password", type="password", value="password123")
            if st.form_submit_button("Login as Employee"):
                if emp_user in employee_creds and emp_pass == employee_creds[emp_user]:
                    st.session_state.logged_in = True; st.session_state.user_type = "Employee"; st.session_state.username = emp_user; st.toast(f"Welcome, {emp_user}!", icon="👋"); st.rerun()
                else: st.error("Invalid username or password")
    with col2:
        with st.form("customer_login"):
            st.subheader("👤 Customer Access Portal")
            cust_user_id = st.text_input("Customer Login ID", value="PriyaS2345")
            cust_pass = st.text_input("Password (use Mobile Number)", type="password", value="+91 9820012345")
            if st.form_submit_button("Login as Customer"):
                if cust_user_id in customer_creds and cust_pass == customer_creds[cust_user_id]:
                    st.session_state.logged_in = True; st.session_state.user_type = "Customer";
                    st.session_state.customer_data = df[df['LoginUserID'] == cust_user_id].iloc[0].to_dict()
                    st.session_state.username = st.session_state.customer_data['FirstName']
                    st.toast(f"Welcome, {st.session_state.username}!", icon="👋"); st.rerun()
                else: st.error("Invalid Login ID or Password")

def show_employee_portal(df, model, model_columns):
    # This function is now much simpler and robust
    st.title(f"🏢 Employee Portal")
    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state.username.capitalize()}!")
        st.markdown("---")
        st.subheader("Your Performance")
        st.metric("Subscriptions Secured (Month)", "22")
        st.metric("Conversion Rate", "18.5%")
        st.progress(0.73, text="Monthly Target (73%)")
        st.markdown("---")
        
        selection = st.radio("Go to", ["📈 Customer Analytics", "👤 Customer 360° View", "🎯 AI Lead Finder", "✨ Festive Offers"])
        st.markdown("---")
        if st.button("Logout"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()
    
    # Call the selected page function with all required arguments
    if selection == "📈 Customer Analytics": page_analytics(df)
    elif selection == "👤 Customer 360° View": page_customer_360(df, model, model_columns)
    elif selection == "🎯 AI Lead Finder": page_lead_finder(df, model, model_columns)
    elif selection == "✨ Festive Offers": page_bank_offers()

def show_customer_portal():
    # This function is now much simpler
    st.title(f"👤 Customer Portal")
    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state.username}!")
        st.markdown("---")
        selection = st.radio("Go to", ["🏠 Account Summary", "💳 Cards & Loans", "💹 Investment Hub", "🧮 Financial Calculators"])
        st.markdown("---")
        if st.button("Logout"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()

    if selection == "🏠 Account Summary": page_account_summary()
    elif selection == "💳 Cards & Loans": page_cards_and_loans()
    elif selection == "💹 Investment Hub": page_investments()
    elif selection == "🧮 Financial Calculators": page_calculators()

# --- Main App ---
def main():
    # Load CSS at the very beginning
    load_css("style.css")

    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    
    DATA_PATH = "data/bank_data_final.csv"
    df = load_data(DATA_PATH)
    
    # Apply theme class and theme toggle
    theme_class = "dark-mode" if st.session_state.get('theme', 'light') == 'dark' else 'light-mode'
    st.markdown(f'<div class="main-container {theme_class}">', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("---")
        if st.toggle('🌙 Dark Mode', value=(st.session_state.get('theme', 'light') == 'dark')):
            if st.session_state.get('theme') != 'dark':
                st.session_state.theme = 'dark'
                st.rerun()
        else:
            if st.session_state.get('theme') != 'light':
                st.session_state.theme = 'light'
                st.rerun()

    if df is not None:
        if st.session_state.logged_in:
            if st.session_state.user_type == "Employee":
                model_pipeline, model_columns = train_model(df)
                show_employee_portal(df, model_pipeline, model_columns)
            else: # Customer
                st.session_state.all_customers = df
                show_customer_portal()
        else:
            show_login_page(df)
            
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
