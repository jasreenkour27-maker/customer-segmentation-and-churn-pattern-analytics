import streamlit as st

st.set_page_config(page_title="Customer Churn Prediction", page_icon="🏦")

st.title("🏦 Customer Segmentation and Churn Pattern Analytics")
st.write("Unified Mentor Internship Project")

st.header("Project Summary")

st.write("""
This project analyzes customer data from a European bank to predict customer churn.

### Machine Learning Model
- Logistic Regression

### Dataset
- European_Bank.csv

### Model Accuracy
- 79%

### Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
""")

st.success("Project deployed successfully!")
