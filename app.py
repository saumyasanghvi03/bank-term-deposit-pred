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

def page_ai_copilot(df, model, model_columns):
    st.header("🤖 AI Co-Pilot")
    st.markdown(f"**Friday, October 10, 2025 | 1:09 AM**")
    st.info("Here are your AI-powered priorities for today to maximize efficiency and results.", icon="🚀")

    # Task 1: High-Propensity Lead
    unsubscribed_df = df[df['y'] == 'no'].copy()
    leads_to_predict = unsubscribed_df[model_columns]
    predictions = model.predict_proba(leads_to_predict)[:, 1]
    unsubscribed_df['Subscription Likelihood'] = predictions
    top_lead = unsubscribed_df.sort_values(by='Subscription Likelihood', ascending=False).iloc[0]
    
    with st.container(border=True):
        st.subheader("🎯 High-Propensity Follow-up")
        st.write(f"Contact **{top_lead['FirstName']} {top_lead['LastName']}**. Our AI gives them an **{top_lead['Subscription Likelihood']:.0%} probability** of subscribing to a Term Deposit.")
        st.info(f"**Suggestion:** Pitch the 'Diwali Dhamaka FD' offer, as their balance is high (₹{top_lead['balance']:,}).", icon="💡")

    # Task 2: Churn Prevention
    at_risk_customer = df[df['balance'] < 10000].iloc[0]
    with st.container(border=True):
        st.subheader("🔔 Churn Prevention Alert")
        st.write(f"**{at_risk_customer['FirstName']} {at_risk_customer['LastName']}'s** account balance dropped to **₹{at_risk_customer['balance']:,}**. They are now an at-risk customer.")
        st.warning("**Suggestion:** Call to offer a personal loan or discuss investment options for their remaining funds to show support.", icon="🚨")
    
    # Task 3: Cross-Sell Opportunity
    cross_sell_customer = df[(df['housing'] == 'yes') & (df['y'] == 'no')].iloc[0]
    with st.container(border=True):
        st.subheader("🔗 Cross-Sell Opportunity")
        st.write(f"**{cross_sell_customer['FirstName']} {cross_sell_customer['LastName']}** has an active home loan with us but hasn't invested in a term deposit.")
        st.success("**Suggestion:** Offer them a small, high-yield Fixed Deposit as a secure investment alongside their loan.", icon="🤝")

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

def page_account_summary():
    customer_data = st.session_state.customer_data
    zen_mode = st.session_state.get('zen_mode', False)

    st.header(f"Welcome Back, {customer_data['FirstName']}!")
    
    st.subheader("Your Account Details")
    col1, col2 = st.columns(2)
    with col1: st.text_input("Account Number", value=customer_data['AccountNumber'], disabled=True)
    with col2: st.text_input("IFSC Code", value=customer_data['IFSCCode'], disabled=True)
    
    st.subheader("Account Balances")
    if zen_mode:
        st.info("🧘 Zen Mode is active. Balances are hidden to promote financial peace of mind.", icon="✨")
        cols = st.columns(len(st.session_state.accounts))
        affirmations = ["You're on track!", "Your savings are growing.", "Keep up the great work!"]
        for i, acc_name in enumerate(st.session_state.accounts.keys()):
            cols[i].metric(acc_name, affirmations[i % len(affirmations)])
    else:
        cols = st.columns(len(st.session_state.accounts))
        for i, (acc_name, acc_balance) in enumerate(st.session_state.accounts.items()): cols[i].metric(acc_name, f"₹{acc_balance:,.2f}")
    
    st.markdown("---")
    if zen_mode:
        st.warning("Quick actions are disabled in Zen Mode to prevent impulsive decisions. Toggle Zen Mode off in the sidebar to proceed.", icon="🧘")
    else:
        st.subheader("Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("📲 Send Money via UPI"):
                with st.form("upi_form", clear_on_submit=True):
                    recipient_upi_id = st.text_input("Recipient UPI ID", "merchant@okbank")
                    amount = st.number_input("Amount (₹)", min_value=1.0, step=10.0)
                    debit_account = st.selectbox("From Account", list(st.session_state.accounts.keys()), key="upi_debit")
                    
                    # UPI Pay Later Feature
                    use_credit = st.checkbox("Pay using your UPI Credit Line")
                    if use_credit:
                        available_credit = st.session_state.card_details['limit'] - st.session_state.card_details['outstanding']
                        st.info(f"Available Credit: ₹{available_credit:,.2f}")

                    if st.form_submit_button("Send via UPI"):
                        if use_credit:
                            if amount > available_credit: st.error("Insufficient credit limit.")
                            else:
                                st.session_state.card_details['outstanding'] += amount
                                new_tx = {"Date": datetime.now().strftime('%Y-%m-%d'), "Description": f"UPI on Credit to {recipient_upi_id}", "Amount (₹)": -amount, "Category": "Credit Spends"}
                                st.session_state.transactions.insert(0, new_tx)
                                st.toast(f"✅ ₹{amount} paid on credit!", icon="💳"); st.rerun()
                        else:
                            if amount > st.session_state.accounts[debit_account]: st.error("Insufficient balance.")
                            else:
                                st.session_state.accounts[debit_account] -= amount
                                new_tx = {"Date": datetime.now().strftime('%Y-%m-%d'), "Description": f"UPI to {recipient_upi_id}", "Amount (₹)": -amount, "Category": "Payments"}
                                st.session_state.transactions.insert(0, new_tx)
                                st.toast(f"✅ ₹{amount} sent successfully!", icon="🎉"); st.rerun()
        with col2:
            with st.expander("🏦 Within-Bank Transfer"):
                with st.form("transfer_form", clear_on_submit=True):
                    # ... (form code is unchanged)
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
    # ... (code for this page is unchanged)
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
                if st.button("Deactivate Round-Up Bot"): st.session_state.bots["round_up"] = False; st.toast("Round-Up Bot deactivated.", icon="⏸️"); st.rerun()
            else:
                if st.button("Activate Round-Up Bot"): st.session_state.bots["round_up"] = True; st.toast("Round-Up Bot activated!", icon="🚀"); st.rerun()
        with col2:
            st.write(""); st.write("")
            if is_active: st.success("✅ ACTIVE")
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
    # ... (code for this page is unchanged)
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

def page_financial_health():
    # ... (code for this page is unchanged)
    st.header("❤️ Automatic Financial Health Analysis")
    st.markdown("Our AI automatically analyzes your profile to generate your financial health score and personalized recommendations.")
    customer_data = st.session_state.customer_data
    score = 0; pro_tips = []
    balance = sum(st.session_state.accounts.values())
    if balance > 500000: score += 40; pro_tips.append("Your savings are excellent! Consider moving surplus cash to investments for better growth.")
    elif balance > 200000: score += 30; pro_tips.append("You have a good savings base. It's a great time to start a goal-based SIP.")
    elif balance > 50000: score += 20; pro_tips.append("You're on the right track! Focus on building an emergency fund covering 3-6 months of expenses.")
    else: score += 10; pro_tips.append("Your top priority should be to build a consistent saving habit. Start with a small recurring deposit.")
    if customer_data['loan'] == 'no' and customer_data['housing'] == 'no': score += 30
    elif customer_data['loan'] == 'yes' and customer_data['housing'] == 'yes': score += 10; pro_tips.append("Managing multiple loans can be challenging. Consider strategies for debt consolidation or prepayment.")
    else: score += 20; pro_tips.append("You are managing your loans well. Ensure you are paying your EMIs on time to maintain a good credit score.")
    if any(goal['invested'] > 0 for goal in st.session_state.goals) or st.session_state.bots['round_up_pot'] > 0: score += 30
    else: score += 10; pro_tips.append("You have not yet started investing. Activate our 'Algo Savings' bots to begin your investment journey with small, automated steps.")
    st.subheader("Your Financial Health Score")
    col1, col2 = st.columns([1,2])
    with col1:
        st.metric("Score", f"{score:.0f} / 100")
        if score > 80: st.success("Status: Excellent")
        elif score > 50: st.warning("Status: Good")
        else: st.error("Status: Needs Attention")
    with col2:
        if score > 80: st.markdown(f'<div style="width: 100%; background-color: #ddd; border-radius: 10px;"><div style="width: {score}%; background-color: #28a745; text-align: right; color: white; padding:5px; border-radius: 10px;"><b>{score}%</b></div></div>', unsafe_allow_html=True)
        elif score > 50: st.markdown(f'<div style="width: 100%; background-color: #ddd; border-radius: 10px;"><div style="width: {score}%; background-color: #ffc107; text-align: right; color: black; padding:5px; border-radius: 10px;"><b>{score}%</b></div></div>', unsafe_allow_html=True)
        else: st.markdown(f'<div style="width: 100%; background-color: #ddd; border-radius: 10px;"><div style="width: {score}%; background-color: #dc3545; text-align: right; color: white; padding:5px; border-radius: 10px;"><b>{score}%</b></div></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("💡 AI-Powered Pro-Tips")
    for tip in pro_tips[:3]: st.info(tip, icon="🧠")

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
        selection = st.radio("Go to", ["🤖 AI Co-Pilot", "📈 Customer Analytics", "👤 Customer 360° View", "🎯 AI Lead Finder", "✨ Festive Offers"])
        st.markdown("---")
        if st.button("Logout"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()
    if selection == "🤖 AI Co-Pilot": page_ai_copilot(df, model, model_columns)
    elif selection == "📈 Customer Analytics": page_analytics(df)
    elif selection == "👤 Customer 360° View": page_customer_360(df, model, model_columns)
    elif selection == "🎯 AI Lead Finder": page_lead_finder(df, model, model_columns)
    elif selection == "✨ Festive Offers": page_bank_offers()

def show_customer_portal():
    st.title(f"👤 Customer Portal")
    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state.username}!")
        st.markdown("---")
        selection = st.radio("Go to", ["🏠 Account Summary", "🤖 Algo Savings", "💳 Cards & Loans", "💹 Investment Hub", "🧮 Financial Calculators", "❤️ Financial Health"])
        st.markdown("---")
        # Zen Mode Toggle
        st.session_state.zen_mode = st.toggle('🧘 Zen Mode', value=st.session_state.get('zen_mode', False), help="Hide balances and disable transactions for a stress-free experience.")
        st.markdown("---")
        if st.button("Logout"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()
    if selection == "🏠 Account Summary": page_account_summary()
    elif selection == "🤖 Algo Savings": page_algo_bots()
    elif selection == "💳 Cards & Loans": page_cards_and_loans()
    elif selection == "💹 Investment Hub": page_investments()
    elif selection == "🧮 Financial Calculators": page_calculators()
    elif selection == "❤️ Financial Health": page_financial_health()

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
