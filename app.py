import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="FinanSage AI Portal",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Asset Caching (No changes needed here) ---
@st.cache_data
def load_data(url):
    try:
        raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        return pd.read_csv(raw_url)
    except Exception as e:
        st.error(f"Error loading data from URL: {e}")
        return None

@st.cache_resource
def train_model(df):
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

# --- Employee Portal Pages (No changes needed here) ---
def page_analytics(df):
    st.header("📊 Customer Analytics Dashboard")
    st.markdown("An in-depth look into the bank's customer base.")
    
    st.subheader("Key Performance Indicators (KPIs)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{df.shape[0]:,}")
    subscription_rate = df['y'].value_counts(normalize=True).get('yes', 0) * 100
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

def page_prediction(df, model_pipeline):
    st.header("🔮 Subscription Propensity AI")
    st.markdown("Predict a customer's likelihood to subscribe to a term deposit.")
    
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
            balance = st.number_input("Account Balance (€)", -10000, 150000, 1500)
            housing = st.selectbox("Has Housing Loan?", ["no", "yes"])
            loan = st.selectbox("Has Personal Loan?", ["no", "yes"])
            campaign = st.number_input("Number of Contacts in Campaign", 1, 100, 1)

        submitted = st.form_submit_button("🧠 Predict Likelihood")

    if submitted:
        input_data = pd.DataFrame({
            'age': [age], 'job': [job], 'marital': [marital], 'education': [education],
            'balance': [balance], 'housing': [housing], 'loan': [loan], 'campaign': [campaign]
        })
        prediction_proba = model_pipeline.predict_proba(input_data)[0][1]
        st.subheader("Prediction Result")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Subscription Probability", f"{prediction_proba:.1%}")
            if prediction_proba > 0.5: st.success("High Likelihood to Subscribe")
            else: st.error("Low Likelihood to Subscribe")
        with col2:
            st.progress(prediction_proba)
            st.markdown(f"There is a **{prediction_proba:.1%}** probability that this customer will subscribe.")

def page_bank_offers():
    st.header("🎁 Customer Offers & Promotions")
    st.markdown("Present these exclusive offers to eligible customers.")
    offers = [
        {"title": "Platinum Home Loan Offer", "icon": "🏡", "rate": "6.75%", "benefit": "Zero Processing Fees", "description": "A limited-time offer for new home buyers with a CIBIL score above 750."},
        {"title": "Millennial Savings Account", "icon": "📱", "rate": "4.5%", "benefit": "Free Online Trading Account", "description": "A high-interest digital savings account for customers aged 21-35."},
        {"title": "Senior Citizen Gold FD", "icon": "👴👵", "rate": "7.5%", "benefit": "Higher Interest Rate", "description": "An exclusive Fixed Deposit scheme for senior citizens offering an additional 0.5% interest."},
        {"title": "Global Travel Forex Card", "icon": "✈️", "rate": "N/A", "benefit": "Zero Currency Markup", "description": "Load up to 15 currencies and enjoy zero markup on all international transactions."}
    ]
    for offer in offers:
        st.markdown(f"""
        <div style="border: 2px solid #2E86C1; border-radius: 10px; padding: 15px; margin-bottom: 20px; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);">
            <h3>{offer['icon']} {offer['title']}</h3>
            <p><strong>Key Benefit:</strong> <span style="color: #2ECC71;">{offer['benefit']}</span> | <strong>Interest Rate:</strong> {offer['rate']}</p>
            <p>{offer['description']}</p>
        </div>
        """, unsafe_allow_html=True)

# --- NEW/UPDATED Customer Portal Pages ---

def page_account_summary():
    st.header(f"Welcome Back, {st.session_state.username.capitalize()}!")
    st.markdown("Here is a summary of your accounts and recent activity.")

    if 'accounts' not in st.session_state:
        st.session_state.accounts = {"Checking": 12540.50, "Savings": 7850.25}

    st.subheader("Account Balances")
    col1, col2 = st.columns(2)
    col1.metric("Checking Account", f"€{st.session_state.accounts['Checking']:,.2f}")
    col2.metric("Savings Account", f"€{st.session_state.accounts['Savings']:,.2f}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Recent Transactions")
        transactions = {
            "Date": ["2025-10-09", "2025-10-08", "2025-10-07", "2025-10-05"],
            "Description": ["Online Shopping - Amazon", "Grocery Store - BigBazaar", "Salary Credit", "Utility Bill - Electricity"],
            "Amount (€)": [-150.75, -88.20, 5000.00, -120.00]
        }
        st.dataframe(pd.DataFrame(transactions), use_container_width=True)
    with col2:
        st.subheader("Quick Actions")
        with st.expander("💸 Make a Deposit"):
            account_to_deposit = st.selectbox("Select Account", list(st.session_state.accounts.keys()))
            deposit_amount = st.number_input("Deposit Amount (€)", 10.0, 10000.0, 50.0, 10.0, key="deposit")
            if st.button("Confirm Deposit"):
                st.session_state.accounts[account_to_deposit] += deposit_amount
                st.success(f"Deposit successful! New balance: €{st.session_state.accounts[account_to_deposit]:,.2f}")
                st.rerun()

def page_investments():
    st.header("💹 Investment Hub")
    st.markdown("Explore curated investment opportunities for 2025. *For demonstration purposes only. Not financial advice.*")
    # ... (code for this page remains the same as previous version)
    mf_data = [
        {"name": "Nifty 50 Index Fund", "category": "Index Fund", "risk": "Moderate", "desc": "Invests in the top 50 Indian companies. Ideal for stable, long-term growth reflecting market performance."},
        {"name": "ELSS Tax Saver Fund", "category": "Tax Saver (ELSS)", "risk": "Moderately High", "desc": "Offers tax benefits under Section 80C with a 3-year lock-in. A great tool for wealth creation and tax saving."},
        {"name": "Small Cap Momentum Fund", "category": "Small Cap", "risk": "Very High", "desc": "Invests in high-growth small-cap companies. Suitable for aggressive investors with a long investment horizon."},
    ]
    etf_data = [
        {"name": "Nifty 50 ETF", "category": "Equity Index", "risk": "Moderate", "desc": "Tracks the Nifty 50 index, offering diversified exposure to large-cap stocks at a very low cost."},
        {"name": "Gold ETF", "category": "Commodity", "risk": "Low to Moderate", "desc": "Invests in physical gold. A great way to hedge against inflation and market volatility."},
        {"name": "IT Sector ETF", "category": "Sectoral", "risk": "High", "desc": "Focuses on top Indian IT companies. Ideal for investors bullish on the growth of the technology sector."}
    ]
    tab1, tab2 = st.tabs(["Mutual Funds (SIP)", "Exchange-Traded Funds (ETFs)"])
    with tab1:
        st.subheader("Top Mutual Funds to SIP in 2025")
        for mf in mf_data:
            with st.container(border=True):
                st.markdown(f"**{mf['name']}**")
                st.markdown(f"*{mf['category']}* | **Risk:** `{mf['risk']}`")
                st.write(mf['desc'])
    with tab2:
        st.subheader("Top ETFs to Buy in 2025")
        for etf in etf_data:
            with st.container(border=True):
                st.markdown(f"**{etf['name']}**")
                st.markdown(f"*{etf['category']}* | **Risk:** `{etf['risk']}`")
                st.write(etf['desc'])

def page_calculators():
    st.header("🧮 Financial Calculators")
    st.markdown("Plan your financial future with our suite of powerful calculators.")
    
    tab1, tab2, tab3 = st.tabs(["SIP Calculator", "Loan EMI Calculator", "Retirement Planner"])

    with tab1:
        st.subheader("Systematic Investment Plan (SIP) Calculator")
        monthly_investment = st.slider("Monthly Investment (€)", 50, 5000, 500)
        expected_return = st.slider("Expected Annual Return (%)", 1.0, 30.0, 12.0, 0.5)
        investment_period = st.slider("Investment Period (Years)", 1, 30, 10)
        
        invested_amount = monthly_investment * investment_period * 12
        i = (expected_return / 100) / 12
        n = investment_period * 12
        future_value = monthly_investment * (((1 + i)**n - 1) / i) * (1 + i)
        
        col1, col2 = st.columns(2)
        col1.metric("Total Invested Amount", f"€{invested_amount:,.0f}")
        col2.metric("Projected Future Value", f"€{future_value:,.0f}")

    with tab2:
        st.subheader("Equated Monthly Instalment (EMI) Calculator")
        loan_amount = st.number_input("Loan Amount (€)", 1000, 10000000, 50000)
        interest_rate = st.slider("Annual Interest Rate (%)", 1.0, 20.0, 8.5, 0.1)
        loan_tenure = st.slider("Loan Tenure (Years)", 1, 30, 5)

        r = (interest_rate / 100) / 12
        n = loan_tenure * 12
        emi = (loan_amount * r * (1 + r)**n) / ((1 + r)**n - 1)
        total_payment = emi * n

        col1, col2 = st.columns(2)
        col1.metric("Monthly EMI Payment", f"€{emi:,.2f}")
        col2.metric("Total Payment (Principal + Interest)", f"€{total_payment:,.0f}")
        
    with tab3:
        st.subheader("Retirement Corpus Planner")
        current_age = st.slider("Your Current Age", 18, 60, 30)
        retirement_age = st.slider("Target Retirement Age", 50, 70, 60)
        monthly_expenses = st.number_input("Current Monthly Expenses (€)", 100, 10000, 1000)
        expected_inflation = st.slider("Expected Inflation Rate (%)", 1.0, 10.0, 6.0, 0.5)

        years_to_retire = retirement_age - current_age
        future_monthly_expenses = monthly_expenses * (1 + expected_inflation / 100)**years_to_retire
        retirement_corpus = future_monthly_expenses * 12 * 25 # Using a simple 4% withdrawal rule

        st.metric("Estimated Retirement Corpus Needed", f"€{retirement_corpus:,.0f}")
        st.info(f"You will need approximately €{future_monthly_expenses:,.0f} per month at retirement. The corpus is calculated to support this lifestyle.")

def page_health_check():
    st.header("❤️ Financial Health Check")
    st.markdown("Answer a few questions to get your financial health score and personalized tips.")

    with st.form("health_check_form"):
        st.subheader("Your Financial Habits")
        q1 = st.radio("How much of your monthly income do you save?", ["Less than 10%", "10% - 20%", "20% - 30%", "More than 30%"], index=1)
        q2 = st.radio("Do you have an emergency fund covering 3-6 months of expenses?", ["No", "Partially", "Yes"], index=1)
        q3 = st.radio("How do you manage your credit card debt?", ["I don't have a credit card", "I pay the minimum due", "I pay in full every month"], index=2)
        q4 = st.radio("Do you have health and life insurance coverage?", ["None", "Only one", "Both"], index=1)

        submitted = st.form_submit_button("Calculate My Score")
        if submitted:
            score = 0
            score += {"Less than 10%": 1, "10% - 20%": 2, "20% - 30%": 3, "More than 30%": 4}[q1]
            score += {"No": 1, "Partially": 2, "Yes": 3}[q2]
            score += {"I don't have a credit card": 3, "I pay the minimum due": 1, "I pay in full every month": 4}[q3]
            score += {"None": 1, "Only one": 2, "Both": 3}[q4]
            
            total_score = (score / 14) * 100 # Max score is 14
            st.subheader("Your Financial Health Score")
            st.metric("Score", f"{total_score:.0f} / 100")
            st.progress(int(total_score))
            
            if total_score > 80:
                st.success("Excellent! You have strong financial habits. Keep it up!")
            elif total_score > 50:
                st.warning("Good, but there's room for improvement. Focus on building your emergency fund and increasing savings.")
            else:
                st.error("Needs Attention. It's time to prioritize creating a budget and a plan for savings and insurance.")


# --- Login & Portal Logic ---
def show_login_page():
    # ... (code for this function remains the same)
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
    # ... (code for this function remains the same, just added 'Customer Offers')
    with st.sidebar:
        st.markdown(f"### Welcome, {st.session_state.username.capitalize()}!")
        st.markdown("---")
        page_options = {
            "📈 Customer Analytics": lambda: page_analytics(df),
            "🔮 Propensity AI": lambda: page_prediction(df, model),
            "🎁 Customer Offers": page_bank_offers
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
            "💹 Investment Hub": page_investments,
            "🧮 Financial Calculators": page_calculators,
            "❤️ Financial Health Check": page_health_check
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
