import os
import pandas as pd
import streamlit as st
from streamlit_js_eval import streamlit_js_eval as sje

st.header("HN Score Prediction")

with st.form('form'):
    title = st.text_input("Title Here:")
    link = st.text_input("Link to Post")
    username = st.text_input("Enter Username")
    submit = st.form_submit_button("Submit")

if submit:
    st.write('form submitted')
    st.json({
        "title": title,
        "link": link,
        "username": username
    })
    
    # result = 
    # st.subheader("Model Output:")
    # st.json(result)