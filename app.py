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

# --- Robust File Loading Functions ---
@st.cache_data
def load_css(file_name):
    """Loads a CSS file from the same directory as the script."""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            st.markdown('<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Error: The 'style.css' file was not found. Please ensure it is in the main project directory.")

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
                model = st.session_state.model; model_columns = st.session_state.model_columns
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

def page_account_summary():
    customer_data = st.session_state.customer_data
    st.header(f"Welcome Back, {customer_data['FirstName']}!")
    st.subheader("Your Account Details")
    col1, col2 = st.columns(2)
    with col1: st.text_input("Account Number", value=customer_data['AccountNumber'], disabled=True)
    with col2: st.text_input("IFSC Code", value=customer_data['IFSCCode'], disabled=True)
    st.subheader("Account Balances")
    cols = st.columns(len(st.session_state.accounts))
    for i, (acc_name, acc_balance) in enumerate(st.session_state.accounts.items()): cols[i].metric(acc_name, f"₹{acc_balance:,.2f}")
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
    st.subheader("My Bot Portfolio")
    with st.container(border=True):
        total_invested = st.session_state.bots['round_up_pot'] + sum(g['invested'] for g in st.session_state.goals)
        total_value = st.session_state.bots['round_up_value'] + sum(g['value'] for g in st.session_state.goals)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Amount Invested", f"₹{total_invested:,.2f}")
        col2.metric("Current Portfolio Value", f"₹{total_value:,.2f}")
        if st.button("Simulate 1 Month of Investing"):
            st.session_state.bots['round_up_pot'] += random.uniform(150, 400)
            st.session_state.bots['round_up_value'] = st.session_state.bots['round_up_pot'] * random.uniform(1.01, 1.03)
            for goal in st.session_state.goals:
                goal['invested'] += goal['sip']
                goal['value'] += goal['sip'] * random.uniform(1.0, 1.05)
            st.toast("Simulated one month of automated investing!", icon="📈"); st.rerun()
        st.markdown("---")
        if st.session_state.bots['round_up']:
            st.write(f"💰 **Round-Up Savings (Liquid Fund):** Current Value **₹{st.session_state.bots['round_up_value']:,.2f}**")
        for goal in st.session_state.goals:
            progress = min(goal['value'] / goal['target'], 1.0) if goal['target'] > 0 else 0
            st.write(f"🎯 **Goal: {goal['name']}** - Current Value **₹{goal['value']:,.2f}** / ₹{goal['target']:,}")
            st.progress(progress)
    st.markdown("---")
    st.subheader("Activate & Manage Bots")
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("💰 Round-Up Savings Bot")
            st.write("Automatically rounds up your daily spends and invests the change.")
            is_active = st.session_state.bots["round_up"]
            if is_active:
                if st.button("Deactivate Round-Up Bot"):
                    st.session_state.bots["round_up"] = False; st.toast("Round-Up Bot deactivated.", icon="⏸️"); st.rerun()
            else:
                if st.button("Activate Round-Up Bot"):
                    st.session_state.bots["round_up"] = True; st.toast("Round-Up Bot activated!", icon="🚀"); st.rerun()
        with col2:
            st.write(""); st.write("") # Spacer
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
        with col2:
            if st.button("🚀 Start this SIP Plan", use_container_width=True):
                new_goal = {"name": goal, "target": target_amount, "sip": monthly_sip, "invested": 0, "value": 0}
                st.session_state.goals.append(new_goal)
                st.success(f"Your SIP for '{goal}' is now active and tracked in your portfolio!"); st.balloons(); st.rerun()

def page_cards_and_loans():
    st.header("💳 Cards & Loans")
    st.subheader("Your Credit Card Summary")
    card = st.session_state.card_details
    col1, col2, col3 = st.columns(3)
    col1.metric("Credit Limit", f"₹{card['limit']:,.2f}")
    col2.metric("Outstanding Amount", f"₹{card['outstanding']:,.2f}")
    utilization = (card['outstanding'] / card['limit']) if card['limit'] > 0 else 0
    col3.metric("Credit Utilization", f"{utilization:.1%}")
    st.progress(utilization)
    if card['outstanding'] > 0.01:
        with st.form("card_payment_form"):
            st.subheader("Make a Card Payment")
            payment_amount = st.number_input("Amount to Pay (₹)", min_value=0.01, max_value=card['outstanding'], value=card['outstanding'])
            payment_account = st.selectbox("Pay from Account", list(st.session_state.accounts.keys()))
            if st.form_submit_button("Pay Credit Card Bill"):
                if payment_amount > st.session_state.accounts[payment_account]: st.error("Insufficient balance.")
                else:
                    st.session_state.accounts[payment_account] -= payment_amount
                    st.session_state.card_details['outstanding'] -= payment_amount
                    new_tx = {"Date": datetime.now().strftime('%Y-%m-%d'), "Description": "Credit Card Bill Payment", "Amount (₹)": -payment_amount, "Category": "Bills"}
                    st.session_state.transactions.insert(0, new_tx)
                    st.toast("✅ Card payment successful!", icon="💳"); st.rerun()
    else: st.success("🎉 Your credit card bill is fully paid!")

def page_investments():
    st.header("💹 Investment Hub")
    mf_data = [{"name": "Parag Parikh Flexi Cap Fund", "category": "Flexi Cap", "risk": "Moderately High", "desc": "A popular choice for its diversified portfolio across domestic and international equities."}, {"name": "SBI Contra ESG Fund", "category": "Thematic - ESG", "risk": "High", "desc": "Invests in companies with strong Environmental, Social, and Governance (ESG) scores, following a contrarian strategy."}, {"name": "Quant Small Cap Fund", "category": "Small Cap", "risk": "Very High", "desc": "Known for its aggressive, high-growth strategy in the small-cap segment, suitable for high-risk investors."}]
    etf_data = [{"name": "Nifty 50 BEES ETF", "category": "Index", "risk": "Moderate", "desc": "Tracks the Nifty 50 index, offering a simple, low-cost way to invest in India's top companies."}, {"name": "Mirae Asset Nifty EV & New Age Automotive ETF", "category": "Thematic", "risk": "High", "desc": "Provides exposure to the rapidly growing Electric Vehicle and new-age automotive technology sectors."}, {"name": "ICICI Prudential Silver ETF", "category": "Commodity", "risk": "High", "desc": "Invests in physical silver, offering a hedge against inflation and a play on industrial demand."}]
    tab1, tab2 = st.tabs(["Mutual Funds (SIP)", "Exchange-Traded Funds (ETFs)"])
    with tab1:
        st.subheader("Top Mutual Funds for SIP in 2025")
        for mf in mf_data:
            with st.container(border=True): st.markdown(f"**{mf['name']}**\n\n*{mf['category']}* | **Risk:** `{mf['risk']}`\n\n{mf['desc']}")
    with tab2:
        st.subheader("Top ETFs to Buy in 2025")
        for etf in etf_data:
            with st.container(border=True): st.markdown(f"**{etf['name']}**\n\n*{etf['category']}* | **Risk:** `{etf['risk']}`\n\n{etf['desc']}")

def page_calculators():
    st.header("🧮 Financial Calculators")
    tab1, tab2, tab3 = st.tabs(["SIP Calculator", "Loan EMI Calculator", "Retirement Planner"])
    with tab1:
        st.subheader("Systematic Investment Plan (SIP) Calculator")
        monthly_investment = st.slider("Monthly Investment (₹)", 1000, 100000, 5000, key="sip_inv")
        expected_return = st.slider("Expected Annual Return (%)", 1.0, 30.0, 12.0, 0.5, key="sip_ret")
        investment_period = st.slider("Investment Period (Years)", 1, 30, 10, key="sip_yrs")
        invested_amount = monthly_investment * investment_period * 12
        i = (expected_return / 100) / 12
        n = investment_period * 12
        future_value = monthly_investment * (((1 + i)**n - 1) / i) * (1 + i)
        col1, col2 = st.columns(2)
        col1.metric("Total Invested Amount", f"₹{invested_amount:,.0f}")
        col2.metric("Projected Future Value", f"₹{future_value:,.0f}")
    with tab2:
        st.subheader("Equated Monthly Instalment (EMI) Calculator")
        loan_amount = st.number_input("Loan Amount (₹)", 10000, 10000000, 500000)
        interest_rate = st.slider("Annual Interest Rate (%)", 1.0, 20.0, 8.5, 0.1)
        loan_tenure = st.slider("Loan Tenure (Years)", 1, 30, 5)
        r = (interest_rate / 100) / 12
        n = loan_tenure * 12
        emi = (loan_amount * r * (1 + r)**n) / ((1 + r)**n - 1)
        total_payment = emi * n
        col1, col2 = st.columns(2)
        col1.metric("Monthly EMI Payment", f"₹{emi:,.2f}")
        col2.metric("Total Payment", f"₹{total_payment:,.0f}")
    with tab3:
        st.subheader("Retirement Corpus Planner")
        current_age = st.slider("Your Current Age", 18, 60, 30)
        retirement_age = st.slider("Target Retirement Age", 50, 70, 60)
        monthly_expenses = st.number_input("Current Monthly Expenses (₹)", 5000, 200000, 30000)
        expected_inflation = st.slider("Expected Inflation Rate (%)", 1.0, 10.0, 6.0, 0.5)
        years_to_retire = retirement_age - current_age
        future_monthly_expenses = monthly_expenses * (1 + expected_inflation / 100)**years_to_retire
        retirement_corpus = future_monthly_expenses * 12 * 25
        st.metric("Estimated Retirement Corpus Needed", f"₹{retirement_corpus:,.0f}")

def page_health_check():
    st.header("❤️ Financial Health Check")
    st.markdown("Answer a few questions to get your financial health score and personalized tips.")
    with st.form("health_check_form"):
        st.subheader("Your Financial Habits")
        q1 = st.radio("How much of your monthly income do you save?", ["Less than 10%", "10% - 20%", "20% - 30%", "More than 30%"], index=1)
        q2 = st.radio("Do you have an emergency fund covering 3-6 months of expenses?", ["No", "Partially", "Yes"], index=1)
        q3 = st.radio("How do you manage your credit card debt?", ["I don't have a credit card", "I pay the minimum due", "I pay in full every month"], index=2)
        q4 = st.radio("Do you have health and life insurance coverage?", ["None", "Only one", "Both"], index=1)
        if st.form_submit_button("Calculate My Score"):
            score = 0
            score += {"Less than 10%": 1, "10% - 20%": 2, "20% - 30%": 3, "More than 30%": 4}[q1]
            score += {"No": 1, "Partially": 2, "Yes": 3}[q2]
            score += {"I don't have a credit card": 3, "I pay the minimum due": 1, "I pay in full every month": 4}[q3]
            score += {"None": 1, "Only one": 2, "Both": 3}[q4]
            total_score = (score / 14) * 100
            st.subheader("Your Financial Health Score")
            st.metric("Score", f"{total_score:.0f} / 100")
            st.progress(int(total_score))
            if total_score > 80: st.success("Excellent! You have strong financial habits.")
            elif total_score > 50: st.warning("Good, but there's room for improvement. Focus on building your emergency fund and increasing savings.")
            else: st.error("Needs Attention. It's time to prioritize creating a budget and a plan for savings and insurance.")

# --- Centralized Session State Initialization ---
def initialize_customer_session(customer_data):
    """Initializes all necessary session state variables for a customer login."""
    st.session_state.logged_in = True
    st.session_state.user_type = "Customer"
    st.session_state.customer_data = customer_data
    st.session_state.username = customer_data['FirstName']
    if customer_data['job'] == 'student': st.session_state.accounts = {"Savings": customer_data['balance']}
    else: st.session_state.accounts = {"Checking": customer_data['balance'] * 0.4, "Savings": customer_data['balance'] * 0.6}
    st.session_state.transactions = [
        {"Date": (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'), "Description": "Supermarket", "Amount (₹)": -5210.50, "Category": "Groceries"},
        {"Date": (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d'), "Description": "Salary Credit", "Amount (₹)": 75000.00, "Category": "Income"},
    ]
    st.session_state.bots = {"round_up": False, "smart_transfer": False, "round_up_pot": 0.0, "round_up_value": 0.0}
    st.session_state.goals = []
    st.session_state.card_details = { "limit": 150000, "outstanding": 25800.50 }

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
                        customer_data = df[df['LoginUserID'] == cust_user_id].iloc[0].to_dict()
                        initialize_customer_session(customer_data)
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
