# 🔐 FinanSage AI - Secure Banking Portal

**FinanSage AI** is a comprehensive, AI-powered web application designed for a modern financial institution. This interactive portal provides two distinct experiences based on user roles: a powerful analytics and prediction suite for **Bank Employees** and a user-friendly self-service portal for **Customers**.

Built with Streamlit, this application demonstrates a complete, role-based architecture suitable for real-world financial technology applications, evolving from an initial data analysis dashboard into a multi-functional platform.

**Author:** Saumya Sanghvi, PGDFT2422
**Course:** MMS-FT Sem 3



***

## ✨ Key Features

The portal is divided into two main access levels, each with a tailored set of features.

### 🏢 Bank Employee Portal
An internal suite of tools for data-driven decision-making.
- **Secure Login**: Access restricted to authorized personnel.
- **Customer Analytics Dashboard**: A multi-tab dashboard for a 360-degree view of the customer base, including KPIs, demographic charts, and K-Means customer segmentation.
- **Subscription Propensity AI**: A real-time prediction tool using an **XGBoost** model to score customers on their likelihood to subscribe to a term deposit, enabling smarter marketing campaigns.

### 👤 Customer Access Portal
A clean and simple portal for customers to manage their finances.
- **Secure Customer Login**: Personalized access to account information.
- **Account Summary**: A clear overview of checking and savings account balances.
- **Quick Actions**: Interactive tools to simulate making deposits into their accounts.
- **Financial Tools**: Includes a **Term Deposit Calculator** to help customers plan their investments and visualize growth over time.

***

## 🛠️ Technology Stack

This project leverages a modern, open-source technology stack for data science and web application development:

- **Language**: Python
- **Web Framework**: Streamlit
- **Data Manipulation**: Pandas
- **Machine Learning**: Scikit-learn, XGBoost
- **Data Visualization**: Plotly
- **Model Persistence**: Joblib

***

## 🚀 How to Run the Project

Follow these steps to set up and run the application on your local machine.

### 1. Prerequisites
- Python 3.8+
- A virtual environment (recommended to avoid package conflicts)

### 2. Setup the Environment
Clone the repository, create a virtual environment, and install the required dependencies from the `requirements.txt` file.

```bash
# Clone the project repository
git clone <your-repository-url>
cd <project-directory>

# Create a virtual environment
python -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt
