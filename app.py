import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta, timezone
import os
import random
import time

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

# --- The Main Application Class ---
class FinanSageApp:
    def __init__(self, dataframe):
        self.df = dataframe
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False

    # --- Employee Portal Pages ---
    def page_ai_copilot(self):
        st.header("🤖 AI Co-Pilot")
        st.markdown(f"**Friday, October 10, 2025 | 11:52 AM IST**")
        st.info("Here are your AI-powered priorities for today to maximize efficiency and results.", icon="🚀")
        model = st.session_state.model
        model_columns = st.session_state.model_columns
        unsubscribed_df = self.df[self.df['y'] == 'no'].copy()
        leads_to_predict = unsubscribed_df[model_columns]
        predictions = model.predict_proba(leads_to_predict)[:, 1]
        unsubscribed_df['Subscription Likelihood'] = predictions
        top_lead = unsubscribed_df.sort_values(by='Subscription Likelihood', ascending=False).iloc[0]
        with st.container(border=True):
            st.subheader("🎯 High-Propensity Follow-up")
            st.write(f"Contact **{top_lead['FirstName']} {top_lead['LastName']}**. Our AI gives them an **{top_lead['Subscription Likelihood']:.0%} probability** of subscribing.")
            st.info(f"**Suggestion:** Pitch the 'Diwali Dhamaka FD' offer, as their balance is high (₹{top_lead['balance']:,}).", icon="💡")
        at_risk_customer = self.df[self.df['balance'] < 10000].iloc[0]
        with st.container(border=True):
            st.subheader("🔔 Churn Prevention Alert")
            st.write(f"**{at_risk_customer['FirstName']} {at_risk_customer['LastName']}'s** account balance dropped to **₹{at_risk_customer['balance']:,}**. They are an at-risk customer.")
            st.warning("**Suggestion:** Call to offer a personal loan or discuss investment options to show support.", icon="🚨")
        cross_sell_customer = self.df[(self.df['housing'] == 'yes') & (self.df['y'] == 'no')].iloc[0]
        with st.container(border=True):
            st.subheader("🔗 Cross-Sell Opportunity")
            st.write(f"**{cross_sell_customer['FirstName']} {cross_sell_customer['LastName']}** has an active home loan with us but hasn't invested in a term deposit.")
            st.success("**Suggestion:** Offer them a small, high-yield Fixed Deposit as a secure investment.", icon="🤝")

    def page_analytics(self):
        st.header("📊 Customer Analytics Dashboard")
        st.subheader("Key Performance Indicators (KPIs)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", f"{self.df.shape[0]:,}")
        subscription_rate = self.df['y'].value_counts(normalize=True).get('yes', 0) * 100
        col2.metric("Subscription Rate", f"{subscription_rate:.2f}%")
        avg_balance = self.df['balance'].mean()
        col3.metric("Avg. Balance (₹)", f"{avg_balance:,.0f}")
        avg_age = self.df['age'].mean()
        col4.metric("Avg. Customer Age", f"{avg_age:.1f}")
        st.markdown("---")
        st.subheader("Customer Demographics")
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(px.histogram(self.df, x='age', nbins=40, title='Age Distribution'), use_container_width=True)
        with col2: st.plotly_chart(px.bar(self.df['job'].value_counts().reset_index(), x='job', y='count', title='Job Distribution'), use_container_width=True)

    def page_customer_360(self):
        st.header("👤 Customer 360° View")
        model = st.session_state.model; model_columns = st.session_state.model_columns
        df = self.df
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

    def page_lead_finder(self):
        st.header("🎯 AI Lead Finder")
        st.markdown("A prioritized list of customers with the highest potential to subscribe to a term deposit.")
        model = st.session_state.model; model_columns = st.session_state.model_columns
        unsubscribed_df = self.df[self.df['y'] == 'no'].copy()
        leads_to_predict = unsubscribed_df[model_columns]
        predictions = model.predict_proba(leads_to_predict)[:, 1]
        unsubscribed_df['Subscription Likelihood'] = predictions
        prioritized_leads = unsubscribed_df.sort_values(by='Subscription Likelihood', ascending=False)
        st.dataframe(prioritized_leads[['FirstName', 'LastName', 'MobileNumber', 'age', 'job', 'balance', 'Subscription Likelihood']],
                     use_container_width=True,
                     column_config={"Subscription Likelihood": st.column_config.ProgressColumn("Likelihood", format="%.2f", min_value=0, max_value=1)})

    def page_bank_offers(self):
        st.header("✨ Festive Offers for Diwali 2025 ✨")
        offers = [
            {"title": "Dhanteras Gold Rush", "icon": "🪙", "rate": "Instant 5% Cashback", "benefit": "On Gold Jewellery & Coin Loans", "description": "Celebrate Dhanteras with a personal loan for gold purchases with zero processing fees and 5% cashback on the loan amount."},
            {"title": "Diwali Wheels of Joy", "icon": "🚗", "rate": "Starting at 8.25%", "benefit": "Zero Down Payment on Car Loans", "description": "Our special car loan offer comes with a rock-bottom interest rate and a zero down payment option for approved customers."},
            {"title": "Festive Home Makeover Loan", "icon": "🏡", "rate": "Attractive Low Interest", "benefit": "Quick Personal Loan for Renovations", "description": "Get a quick-disbursal personal loan up to ₹5 Lakhs for home improvements, painting, or buying new appliances."},
            {"title": "Diwali Dhamaka FD", "icon": "💰", "rate": "8.00% p.a.", "benefit": "Special High-Interest Fixed Deposit", "description": "A limited-period Fixed Deposit scheme offering a special high interest rate. Senior citizens get an additional 0.5%!"}
        ]
        for offer in offers:
            st.markdown(f'<div class="offer-card"><h3>{offer["icon"]} {offer["title"]}</h3><p><strong>Key Benefit:</strong> <span style="color: #E67E22; font-weight: bold;">{offer["benefit"]}</span> | <strong>Offer Details:</strong> {offer["rate"]}</p><p>{offer["description"]}</p></div>', unsafe_allow_html=True)
    
    # --- Customer Portal Pages ---
    def page_account_summary(self):
        customer_data = st.session_state.customer_data
        st.header(f"Welcome Back, {customer_data['FirstName']}!")
        current_mode = st.session_state.get('current_mode', 'normal')
        
        mode_details = {
            'commute': {"icon": "🚗", "title": "Your Morning Dash"}, 'lunch': {"icon": "🥗", "title": "Your Lunch Money Roundup"},
            'evening': {"icon": "🌆", "title": "Your Evening Planner"}, 'zen': {"icon": "🧘", "title": "Your Zen Finance View"}
        }
        if current_mode in mode_details and current_mode != 'zen':
            st.subheader(f"{mode_details[current_mode]['icon']} {mode_details[current_mode]['title']}")
            with st.container(border=True):
                if current_mode == 'commute':
                    total_balance = sum(st.session_state.accounts.values())
                    st.metric("Total Account Balance", f"₹{total_balance:,.0f}")
                    st.info("**Tip of the Day:** Even small, consistent investments can lead to significant wealth.", icon="💡")
                elif current_mode == 'lunch':
                    st.write("**1-Minute Challenge:** Can you name the 3 main types of mutual funds?")
                    st.info("**Quick Tip:** Paying your credit card bill in full is the best way to boost your credit score.", icon="💡")
                elif current_mode == 'evening':
                    st.write("A great time to plan for tomorrow. Have you checked your Algo Bot goals?"); st.success("**Featured Read:** [Article] 5 Common Mistakes to Avoid When Investing", icon="📖")
        
        st.subheader("Your Account Details")
        col1, col2 = st.columns(2)
        with col1: st.text_input("Account Number", value=customer_data['AccountNumber'], disabled=True)
        with col2: st.text_input("IFSC Code", value=customer_data['IFSCCode'], disabled=True)
        
        st.subheader("Account Balances")
        if current_mode == 'zen':
            st.info("🧘 Zen Mode is active. Balances are hidden. Quick actions are disabled.", icon="✨")
            cols = st.columns(len(st.session_state.accounts)); affirmations = ["You're on track!", "Your savings are growing."]
            for i, acc_name in enumerate(st.session_state.accounts.keys()): cols[i].metric(acc_name, affirmations[i % len(affirmations)])
        else:
            cols = st.columns(len(st.session_state.accounts))
            for i, (acc_name, acc_balance) in enumerate(st.session_state.accounts.items()): cols[i].metric(acc_name, f"₹{acc_balance:,.2f}")
            st.markdown("---")
            st.subheader("Quick Actions")
            col1, col2 = st.columns(2)
            with col1:
                with st.expander("📲 Send Money via UPI"):
                    with st.form("upi_form", clear_on_submit=True):
                        recipient_upi_id = st.text_input("Recipient UPI ID", "merchant@okbank"); amount = st.number_input("Amount (₹)", min_value=1.0, step=10.0)
                        debit_account = st.selectbox("From Account", list(st.session_state.accounts.keys()), key="upi_debit")
                        if st.form_submit_button("Send via UPI"):
                            if amount > st.session_state.accounts[debit_account]: st.error("Insufficient balance.")
                            else:
                                st.session_state.accounts[debit_account] -= amount; new_tx = {"Date": datetime.now().strftime('%Y-%m-%d, %I:%M %p'), "Description": f"UPI to {recipient_upi_id}", "Amount (₹)": -amount}
                                st.session_state.transactions.insert(0, new_tx); st.toast(f"✅ ₹{amount} sent to {recipient_upi_id}!", icon="🎉"); st.rerun()
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
                                st.session_state.accounts[debit_account] -= amount; new_tx = {"Date": datetime.now().strftime('%Y-%m-%d, %I:%M %p'), "Description": f"Transfer to {recipient_name}", "Amount (₹)": -amount}
                                st.session_state.transactions.insert(0, new_tx); st.toast(f"✅ ₹{amount} transferred to {recipient_name}!", icon="🎉"); st.rerun()
        st.markdown("---")
        st.subheader("Recent Transactions")
        st.dataframe(pd.DataFrame(st.session_state.transactions), use_container_width=True)

    # --- Centralized Session State Initialization ---
    def initialize_customer_session(self, customer_data):
        st.session_state.logged_in = True
        st.session_state.user_type = "Customer"
        st.session_state.customer_data = customer_data
        st.session_state.username = customer_data['FirstName']
        if customer_data['job'] == 'student': st.session_state.accounts = {"Savings": customer_data['balance']}
        else: st.session_state.accounts = {"Checking": customer_data['balance'] * 0.4, "Savings": customer_data['balance'] * 0.6}
        st.session_state.transactions = [
            {"Date": (datetime.now() - timedelta(days=1)).strftime('Yesterday, %I:%M %p'), "Description": "Supermarket", "Amount (₹)": -5210.50, "Category": "Groceries"},
            {"Date": (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d, %I:%M %p'), "Description": "Salary Credit", "Amount (₹)": 75000.00, "Category": "Income"},
        ]
        st.session_state.bots = {"round_up": False, "smart_transfer": False, "round_up_pot": 0.0, "round_up_value": 0.0}
        st.session_state.goals = []
        st.session_state.card_details = { "limit": 150000, "outstanding": 25800.50 }
        st.session_state.notifications = [f"Welcome back, {customer_data['FirstName']}! Your last login was yesterday."]
        st.session_state.last_known_mode = None
    
    # --- Login & Portal Logic ---
    def show_login_page(self):
        st.markdown("<h1 style='text-align: center;'>🔐 FinanSage AI Portal</h1>", unsafe_allow_html=True)
        login_tab, create_account_tab = st.tabs(["Login to Your Account", "Open a New Account"])
        with login_tab:
            col1, col2 = st.columns(2)
            with col1:
                with st.form("employee_login"):
                    st.subheader("🏦 Bank Employee Login")
                    emp_user = st.text_input("Username", value="admin"); emp_pass = st.text_input("Password", type="password", value="password123")
                    if st.form_submit_button("Login as Employee"):
                        employee_creds = {"admin": "password123"}
                        if emp_user in employee_creds and emp_pass == employee_creds[emp_user]:
                            st.session_state.logged_in = True; st.session_state.user_type = "Employee"; st.session_state.username = emp_user; st.toast(f"Welcome, {emp_user}!", icon="👋"); st.rerun()
                        else: st.error("Invalid username or password")
            with col2:
                with st.form("customer_login"):
                    st.subheader("👤 Customer Access Portal")
                    customer_creds = dict(zip(self.df['LoginUserID'], self.df['MobileNumber'].astype(str)))
                    cust_user_id = st.text_input("Customer Login ID", value="PriyaS2345")
                    cust_pass = st.text_input("Password (use Mobile Number)", type="password", value="+91 9820012345")
                    if st.form_submit_button("Login as Customer"):
                        if cust_user_id in customer_creds and cust_pass == customer_creds[cust_user_id]:
                            customer_data = self.df[self.df['LoginUserID'] == cust_user_id].iloc[0].to_dict()
                            self.initialize_customer_session(customer_data)
                            st.toast(f"Welcome, {st.session_state.username}!", icon="👋"); st.rerun()
                        else: st.error("Invalid Login ID or Password")
        with create_account_tab:
            st.subheader("✨ Let's Get You Started")
            with st.form("new_account_form"):
                new_fname = st.text_input("First Name"); new_lname = st.text_input("Last Name")
                new_mobile = st.text_input("Mobile Number (+91)"); new_email = st.text_input("Email Address")
                if st.form_submit_button("Create My Account"):
                    if all([new_fname, new_lname, new_mobile, new_email]):
                        new_cust_id = self.df['CustomerID'].max() + 1; new_acc_num = self.df['AccountNumber'].max() + 1
                        new_login_id = f"{new_fname.capitalize()}{new_lname[0].upper()}{str(new_mobile)[-4:]}"
                        st.success("🎉 Account Created Successfully!")
                        st.markdown("Please use these credentials to log in on the previous tab:")
                        st.text_input("Your New Customer Login ID", value=new_login_id, disabled=True)
                        st.text_input("Your Password (your Mobile Number)", value=new_mobile, disabled=True)
                        st.info("Note: This new account is for simulation only and will not be saved permanently.")
                    else: st.error("Please fill in all the details.")

    def show_employee_portal(self):
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
        if selection == "🤖 AI Co-Pilot": self.page_ai_copilot()
        elif selection == "📈 Customer Analytics": self.page_analytics()
        elif selection == "👤 Customer 360° View": self.page_customer_360()
        elif selection == "🎯 AI Lead Finder": self.page_lead_finder()
        elif selection == "✨ Festive Offers": self.page_bank_offers()

    def show_customer_portal(self):
        st.title(f"👤 Customer Portal")
        ist_time = datetime.now(timezone(timedelta(hours=5, minutes=30)))
        current_hour = ist_time.hour
        mode_names = {'commute': "🚗 Morning Dash", 'lunch': "🥗 Lunch Money", 'evening': "🌆 Evening Planner", 'zen': "🧘 Zen Finance", 'normal': "Normal Mode"}
        if 7 <= current_hour < 12: new_mode = 'commute'
        elif 12 <= current_hour < 15: new_mode = 'lunch'
        elif 17 <= current_hour < 20: new_mode = 'evening'
        elif 22 <= current_hour or current_hour < 7: new_mode = 'zen'
        else: new_mode = 'normal'
        if st.session_state.last_known_mode != new_mode:
            st.toast(f"Switched to {mode_names[new_mode]}!", icon=mode_names[new_mode][0])
            st.session_state.last_known_mode = new_mode
        st.session_state.current_mode = new_mode
        with st.sidebar:
            st.markdown(f"### Welcome, {st.session_state.username}!")
            st.markdown("---")
            notif_count = len(st.session_state.get('notifications', []))
            notif_badge = f" ({notif_count})" if notif_count > 0 else ""
            with st.expander(f"🔔 Notifications{notif_badge}"):
                if notif_count > 0:
                    for notif in st.session_state.notifications: st.info(notif)
                else: st.write("No new notifications.")
            selection = st.radio("Go to", ["🏠 Account Summary", "🤖 Algo Savings", "💳 Cards & Loans", "💹 Investment Hub", "🧮 Financial Calculators", "❤️ Financial Health"])
            st.markdown("---")
            st.session_state.zen_mode = (st.session_state.current_mode == 'zen')
            if st.toggle('🧘 Zen Mode', value=st.session_state.zen_mode, help="Hide balances for a stress-free experience."): st.session_state.current_mode = 'zen'
            st.markdown("---")
            if st.button("Logout"):
                for key in list(st.session_state.keys()): del st.session_state[key]
                st.rerun()
        if selection == "🏠 Account Summary": self.page_account_summary()
        elif selection == "🤖 Algo Savings": self.page_algo_bots()
        elif selection == "💳 Cards & Loans": self.page_cards_and_loans()
        elif selection == "💹 Investment Hub": self.page_investments()
        elif selection == "🧮 Financial Calculators": self.page_calculators()
        elif selection == "❤️ Financial Health": self.page_financial_health()

    def run(self):
        load_css("style.css")
        theme_class = "dark-mode" if st.session_state.get('theme', 'light') == 'dark' else 'light-mode'
        st.markdown(f'<div class="main-container {theme_class}">', unsafe_allow_html=True)
        with st.sidebar:
            st.markdown("---")
            if st.toggle('🌙 Dark Mode', value=(st.session_state.get('theme', 'light') == 'dark')):
                if st.session_state.get('theme') != 'dark': st.session_state.theme = 'dark'; st.rerun()
            else:
                if st.session_state.get('theme') != 'light': st.session_state.theme = 'light'; st.rerun()
            st.markdown("---")
            time_placeholder = st.empty()
        if self.df is not None:
            if st.session_state.logged_in:
                if st.session_state.user_type == "Employee":
                    model_pipeline, model_columns = train_model(self.df)
                    st.session_state.model = model_pipeline; st.session_state.model_columns = model_columns
                    self.show_employee_portal()
                else:
                    st.session_state.all_customers = self.df
                    self.show_customer_portal()
            else:
                self.show_login_page()
        st.markdown('</div>', unsafe_allow_html=True)
        while True:
            ist_time_str = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime('%a, %b %d | %I:%M:%S %p IST')
            time_placeholder.markdown(f"**{ist_time_str}**")
            time.sleep(1)

if __name__ == "__main__":
    DATA_PATH = "data/bank_data_final.csv"
    df = load_data(DATA_PATH)
    if df is not None:
        app = FinanSageApp(df)
        app.run()
