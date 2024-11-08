# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 14:02:44 2024

@author: islam.refaei
"""

# Import Packages
import streamlit as st
import pandas as pd
import joblib


# Load our model pipeline object
model = joblib.load("model.joblib")


# Add title and instructions 
st.title("Purchase Prediction Model")
st.subheader("Enter customer information and submit for likelihood to purchase") 


# age form input

age = st.number_input(
    label= "01- Enter the customer's age",
    min_value= 18,
    max_value= 120,
    value= 35)


# gender form input

gender = st.radio(
    label= "02- Select the customer's gender",
    options = ['M', 'F'])

# credit_score form input

credit_score = st.number_input(
    label= "03- Enter the customer's credit score",
    min_value= 0,
    max_value= 1000,
    value= 500)

# submit button
if st.button('Submit for prediction'):
    
    # Store data in data frame for prediction
    new_data = pd.DataFrame({"age": [age], "gender" : [gender], "credit_score" : [credit_score]})
    
    # apply model pipeline and extract probability
    pred_proba = model.predict_proba(new_data)[0][1]
    
    # output prediction
    st.subheader(f"Baesd on these customer attributes, our model predicts a purchase probabbility of {pred_proba:.0%}") 
