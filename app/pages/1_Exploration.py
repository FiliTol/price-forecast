import streamlit as st
import pandas as pd
import random
from custom.custom_functions import random_color_generator

random.seed(874631)

# The main idea is to create two pages for the app:
# - one for the visualization of current dataset characteristics, providing temporal filters and other info
# - one for the actual prediction, providing empty slots for input data and some KPIs for the results


st.set_page_config(page_title="Exploration")
hide_default_format = """
       <style>
       footer {visibility: hidden;}
       </style>
       """

st.markdown(hide_default_format, unsafe_allow_html=True)
st.title("Exploring the dataset")
