import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="FinanSage AI Portal",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Asset Caching ---
@st.cache_data
def load_data(url):
    """Loads the dataset from the specified GitHub URL with caching."""
    try:
        raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        return pd.read_csv(raw_url)
    except Exception as e:
        st.error(f"Error loading data from URL: {e}")
        return None

@st.cache_resource
def train_model(df):
    """Trains the model on the provided dataframe and returns the pipeline."""
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

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
    with col1:
        st.plotly_chart(px.histogram(df, x='age', nbins=40, title='Age Distribution'), use_container_width=True)
    with col2:
        st.plotly_chart(px.bar(df['job'].value_counts().reset_index(), x='job', y='count', title='Job Distribution'), use_container_width=True)

def page_prediction(df, model_pipeline):
    st.header("🔮 Subscription Propensity AI")
    # ... (code for this page remains the same)
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
            input_data = pd.DataFrame({'age': [age], 'job': [job], 'marital': [marital], 'education': [education], 'balance': [balance], 'housing': [housing], 'loan': [loan], 'campaign': [campaign]})
            prediction_proba = model_pipeline.predict_proba(input_data)[0][1]
            st.subheader("Prediction Result")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Subscription Probability", f"{prediction_proba:.1%}")
                if prediction_proba > 0.5: st.success("High Likelihood")
                else: st.error("Low Likelihood")
            with col2:
                st.progress(prediction_proba)
                st.markdown(f"There is a **{prediction_proba:.1%}** probability that this customer will subscribe.")

def page_bank_offers():
    st.header("✨ Festive Offers for Diwali 2025 ✨")
    # ... (code for this page remains the same)
    st.markdown("Present these exclusive, limited-time offers to eligible customers to celebrate the festive season.")
    offers = [
        {"title": "Dhanteras Gold Rush", "icon": "🪙", "rate": "Instant 5% Cashback", "benefit": "On Gold Jewellery & Coin Loans", "description": "Celebrate Dhanteras by bringing home prosperity. Get an instant personal loan for gold purchases with zero processing fees and receive 5% cashback on the loan amount. Offer valid till Dhanteras evening."},
        {"title": "Diwali Wheels of Joy", "icon": "🚗", "rate": "Starting at 8.25%", "benefit": "Zero Down Payment on Car Loans", "description": "Bring home a new car this Diwali. Our special car loan offer comes with a rock-bottom interest rate and a zero down payment option for approved customers. Includes a complimentary FASTag."},
        {"title": "Festive Home Makeover Loan", "icon": "🏡", "rate": "Attractive Low Interest", "benefit": "Quick Personal Loan for Renovations", "description": "Renovate your home for the festival of lights. Get a quick-disbursal personal loan up to ₹5 Lakhs for home improvements, painting, or buying new appliances. Minimal documentation required."},
        {"title": "Diwali Dhamaka FD", "icon": "💰", "rate": "8.00% p.a.", "benefit": "Special High-Interest Fixed Deposit", "description": "Grow your wealth this Diwali. A limited-period Fixed Deposit scheme for all customers offering a special high interest rate. Senior citizens get an additional 0.5%!"}
    ]
    for offer in offers:
        st.markdown(f"""
        <div style="border: 2px solid #FFC300; border-radius: 10px; padding: 15px; margin-bottom: 20px; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); background-color: #FFF9E6;">
            <h3>{offer['icon']} {offer['title']}</h3>
            <p><strong>Key Benefit:</strong> <span style="color: #E67E22; font-weight: bold;">{offer['benefit']}</span> | <strong>Offer Details:</strong> {offer['rate']}</p>
            <p>{offer['description']}</p>
        </div>
        """, unsafe_allow_html=True)

def page_lead_finder(df, model):
    st.header("🎯 AI Lead Finder")
    st.markdown("A prioritized list of customers with the highest potential to subscribe to a term deposit. Use this list to focus your marketing efforts.")
    
    # Predict on the entire dataset
    unsubscribed_df = df[df['y'] == 'no'].copy()
    predictions = model.predict_proba(unsubscribed_df)[:, 1]
    unsubscribed_df['Subscription Likelihood'] = predictions
    
    # Sort by likelihood
    prioritized_leads = unsubscribed_df.sort_values(by='Subscription Likelihood', ascending=False)
    
    st.dataframe(prioritized_leads[['age', 'job', 'marital', 'balance', 'Subscription Likelihood']],
                 use_container_width=True,
                 column_config={"Subscription Likelihood": st.column_config.ProgressColumn("Likelihood", format="%.2f", min_value=0, max_value=1)})

# --- Customer Portal Pages ---
def page_account_summary():
    st.header(f"Welcome Back, {st.session_state.username.capitalize()}!")

    # Initialize session state for first-time login
    if 'accounts' not in st.session_state:
        st.session_state.accounts = {"Checking": 85450.75, "Savings": 312500.50}
    if 'transactions' not in st.session_state:
        st.session_state.transactions = [
            {"Date": (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'), "Description": "Jewellery Store - Tanishq", "Amount (₹)": -25000.00, "Category": "Shopping"},
            {"Date": (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'), "Description": "Supermarket - Reliance Smart", "Amount (₹)": -5210.50, "Category": "Groceries"},
            {"Date": (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d'), "Description": "Salary Credit", "Amount (₹)": 75000.00, "Category": "Income"},
            {"Date": (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'), "Description": "Zomato Order", "Amount (₹)": -850.00, "Category": "Food"},
            {"Date": (datetime.now() - timedelta(days=6)).strftime('%Y-%m-%d'), "Description": "Utility Bill - Electricity", "Amount (₹)": -3500.00, "Category": "Bills"},
        ]

    st.subheader("Account Balances")
    col1, col2 = st.columns(2)
    col1.metric("Checking Account", f"₹{st.session_state.accounts['Checking']:,.2f}")
    col2.metric("Savings Account", f"₹{st.session_state.accounts['Savings']:,.2f}")

    # Personalized Financial Insights
    savings_balance = st.session_state.accounts['Savings']
    if savings_balance < 50000:
        st.info("💡 **Pro-Tip:** Your savings balance is low. Consider setting up a recurring deposit to build your emergency fund.", icon="🧠")
    elif savings_balance > 500000:
        st.info("💡 **Pro-Tip:** You have a healthy savings balance! Consider exploring our investment options to make your money grow faster.", icon="🧠")

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Spending Habits")
        transactions_df = pd.DataFrame(st.session_state.transactions)
        spending_df = transactions_df[transactions_df['Amount (₹)'] < 0].copy()
        spending_df['Amount (₹)'] = spending_df['Amount (₹)'].abs()
        spending_by_category = spending_df.groupby('Category')['Amount (₹)'].sum().reset_index()
        
        fig = px.pie(spending_by_category, values='Amount (₹)', names='Category', title='Your Recent Spending Breakdown', hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Recent Transactions")
        st.dataframe(pd.DataFrame(st.session_state.transactions).drop(columns=['Category']), use_container_width=True)
    
    # UPI Payment Simulation
    with st.expander("📲 Send Money via UPI"):
        # ... (code for this feature remains the same)
        with st.form("upi_form"):
            recipient_upi_id = st.text_input("Recipient UPI ID", "merchant@okbank")
            amount = st.number_input("Amount (₹)", min_value=1.0, max_value=50000.0, step=10.0)
            remarks = st.text_input("Remarks (Optional)", "Shopping")
            debit_account = st.selectbox("Debit from Account", list(st.session_state.accounts.keys()))
            proceed_to_pay = st.form_submit_button("Proceed to Pay")
            if proceed_to_pay:
                if amount > st.session_state.accounts[debit_account]: st.error("Insufficient balance.")
                else:
                    st.session_state.upi_pin_prompt = True
                    st.session_state.upi_details = {"recipient": recipient_upi_id, "amount": amount, "remarks": remarks, "debit_account": debit_account}
                    st.rerun()

    # UPI PIN Confirmation
    if st.session_state.get('upi_pin_prompt', False):
        details = st.session_state.upi_details
        st.subheader("Confirm Transaction")
        pin = st.text_input("Enter your 4-digit UPI PIN", type="password", max_chars=4)
        if st.button("Confirm Payment"):
            if pin == "1234":
                st.session_state.accounts[details['debit_account']] -= details['amount']
                new_transaction = {"Date": datetime.now().strftime('%Y-%m-%d'), "Description": f"UPI to {details['recipient']} ({details['remarks']})", "Amount (₹)": -details['amount'], "Category": "Transfers"}
                st.session_state.transactions.insert(0, new_transaction)
                st.success("Payment Successful!")
                del st.session_state.upi_pin_prompt; del st.session_state.upi_details
                st.rerun()
            else: st.error("Invalid PIN.")

def page_cards_and_loans():
    st.header("💳 Cards & Loans")

    if 'card_details' not in st.session_state:
        st.session_state.card_details = { "limit": 150000, "outstanding": 25800.50 }

    st.subheader("Your Credit Card Summary")
    card = st.session_state.card_details
    col1, col2, col3 = st.columns(3)
    col1.metric("Credit Limit", f"₹{card['limit']:,.2f}")
    col2.metric("Outstanding Amount", f"₹{card['outstanding']:,.2f}")
    
    utilization = (card['outstanding'] / card['limit']) if card['limit'] > 0 else 0
    col3.metric("Credit Utilization", f"{utilization:.1%}")
    st.progress(utilization)

    with st.form("card_payment_form"):
        st.subheader("Make a Card Payment")
        payment_amount = st.number_input("Amount to Pay (₹)", min_value=100.0, max_value=card['outstanding'], value=card['outstanding'])
        payment_account = st.selectbox("Pay from Account", list(st.session_state.accounts.keys()))
        
        if st.form_submit_button("Pay Credit Card Bill"):
            if payment_amount > st.session_state.accounts[payment_account]:
                st.error("Insufficient balance in the selected account.")
            else:
                st.session_state.accounts[payment_account] -= payment_amount
                st.session_state.card_details['outstanding'] -= payment_amount
                new_transaction = {"Date": datetime.now().strftime('%Y-%m-%d'), "Description": "Credit Card Bill Payment", "Amount (₹)": -payment_amount, "Category": "Bills"}
                st.session_state.transactions.insert(0, new_transaction)
                st.success("Card payment successful!")
                st.rerun()

def page_investments():
    st.header("💹 Investment Hub")
    # ... (code remains same)
    mf_data = [{"name": "Nifty 50 Index Fund", "category": "Index Fund", "risk": "Moderate", "desc": "Invests in India's top 50 companies."}, {"name": "ELSS Tax Saver Fund", "category": "Tax Saver (ELSS)", "risk": "Moderately High", "desc": "Offers tax benefits under Section 80C with a 3-year lock-in."}, {"name": "Gold Fund", "category": "Commodity", "risk": "Low to Moderate", "desc": "A smart way to invest in gold digitally."}]
    etf_data = [{"name": "Nifty 50 ETF", "category": "Equity Index", "risk": "Moderate", "desc": "Tracks the Nifty 50 index at a very low cost."}, {"name": "Gold BEES ETF", "category": "Commodity", "risk": "Low to Moderate", "desc": "Invests in physical gold."}, {"name": "IT BEES ETF", "category": "Sectoral", "risk": "High", "desc": "Focuses on top Indian IT companies."}]
    tab1, tab2 = st.tabs(["Mutual Funds (SIP)", "Exchange-Traded Funds (ETFs)"])
    with tab1:
        for mf in mf_data:
            with st.container(border=True): st.markdown(f"**{mf['name']}**\n\n*{mf['category']}* | **Risk:** `{mf['risk']}`\n\n{mf['desc']}")
    with tab2:
        for etf in etf_data:
            with st.container(border=True): st.markdown(f"**{etf['name']}**\n\n*{etf['category']}* | **Risk:** `{etf['risk']}`\n\n{etf['desc']}")

def page_calculators():
    st.header("🧮 Financial Calculators")
    # ... (code remains same)
    tab1, tab2, tab3 = st.tabs(["SIP Calculator", "Loan EMI Calculator", "Retirement Planner"])
    with tab1:
        st.subheader("Systematic Investment Plan (SIP) Calculator")
        monthly_investment = st.slider("Monthly Investment (₹)", 1000, 100000, 5000)
        expected_return = st.slider("Expected Annual Return (%)", 1.0, 30.0, 12.0, 0.5)
        investment_period = st.slider("Investment Period (Years)", 1, 30, 10)
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

# --- Login & Portal Logic ---
def show_login_page():
    # ... (code remains same)
    st.markdown("<h1 style='text-align: center;'>🔐 FinanSage AI Portal</h1>", unsafe_allow_html=True)
    st.markdown("---")
    employee_creds = {"admin": "password123"}
    customer_creds = {"customer": "mumbai"}
    col1, col2 = st.columns(2)
    with col1:
        with st.form("employee_login"):
            st.subheader("🏦 Bank Employee Login")
            emp_user = st.text_input("Username", key="emp_user", value="admin")
            emp_pass = st.text_input("Password", type="password", key="emp_pass", value="password123")
            if st.form_submit_button("Login as Employee"):
                if emp_user in employee_creds and emp_pass == employee_creds[emp_user]:
                    st.session_state.logged_in = True; st.session_state.user_type = "Employee"; st.session_state.username = emp_user; st.rerun()
                else: st.error("Invalid username or password")
    with col2:
        with st.form("customer_login"):
            st.subheader("👤 Customer Access Portal")
            cust_user = st.text_input("Username", key="cust_user", value="customer")
            cust_pass = st.text_input("Password", type="password", key="cust_pass", value="mumbai")
            if st.form_submit_button("Login as Customer"):
                if cust_user in customer_creds and cust_pass == customer_creds[cust_user]:
                    st.session_state.logged_in = True; st.session_state.user_type = "Customer"; st.session_state.username = cust_user; st.rerun()
                else: st.error("Invalid username or password")

def show_employee_portal(df, model):
    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state.username.capitalize()}!")
        st.markdown("---")
        # Employee Performance Metrics
        st.subheader("Your Performance")
        st.metric("Subscriptions Secured (Month)", "22")
        st.metric("Conversion Rate", "18.5%")
        st.progress(0.73, text="Monthly Target (73%)")
        st.markdown("---")

        page_options = { 
            "📈 Customer Analytics": lambda: page_analytics(df), 
            "🔮 Propensity AI": lambda: page_prediction(df, model), 
            "🎯 AI Lead Finder": lambda: page_lead_finder(df, model),
            "✨ Festive Offers": page_bank_offers
        }
        selection = st.radio("Go to", list(page_options.keys()))
        st.markdown("---")
        if st.button("Logout"):
            for key in st.session_state.keys(): del st.session_state[key]
            st.rerun()
    st.title(f"🏢 Employee Portal: {selection}")
    page_options[selection]()

def show_customer_portal():
    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state.username.capitalize()}!")
        st.markdown("---")
        page_options = { 
            "🏠 Account Summary": page_account_summary, 
            "💳 Cards & Loans": page_cards_and_loans,
            "💹 Investment Hub": page_investments, 
            "🧮 Financial Calculators": page_calculators,
        }
        selection = st.radio("Go to", list(page_options.keys()))
        st.markdown("---")
        if st.button("Logout"):
            for key in st.session_state.keys(): del st.session_state[key]
            st.rerun()
    st.title(f"👤 Customer Portal: {selection}")
    page_options[selection]()

# --- Main App ---
def main():
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    
    DATA_URL = "https://github.com/saumyasanghvi03/bank-term-deposit-pred/blob/main/bank_data.csv"

    if st.session_state.logged_in:
        if st.session_state.user_type == "Employee":
            df = load_data(DATA_URL)
            if df is not None:
                model_pipeline = train_model(df)
                show_employee_portal(df, model_pipeline)
        else:
            show_customer_portal()
    else:
        show_login_page()

if __name__ == "__main__":
    main()
