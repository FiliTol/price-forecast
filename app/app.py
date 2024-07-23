import streamlit as st
import pandas as pd
import random

random.seed(874631)

# The main idea is to create two pages for the app:
# - one for the visualization of current dataset characteristics, providing temporal filters and other info
# - one for the actual prediction, providing empty slots for input data and some KPIs for the results


st.set_page_config(page_title="Price Prediction App")
hide_default_format = """
       <style>
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)
st.title("My first dashboard")


def random_color_generator():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return r, g, b


random_color = random_color_generator()

df = pd.read_csv("data/2023dic/d_listings.csv")
graph = df[['id', 'host_id', 'latitude', 'longitude', 'neighbourhood_cleansed', 'host_response_time']]

zona = st.sidebar.selectbox(
    "Which sestiere do you want to see?",
    graph['neighbourhood_cleansed'].unique().tolist()
)

response_time = st.sidebar.radio('Host answers...',
                                 options=df["host_response_time"].unique().tolist(),
                                 horizontal=True)

col_zona = {}
for i in graph['neighbourhood_cleansed'].unique():
    col_zona[i] = random_color_generator()

graph['color'] = graph['neighbourhood_cleansed'].apply(lambda x: col_zona.get(x))

st.map(graph[(graph['neighbourhood_cleansed'] == zona)], # & (graph['host_response_time'] == response_time)],
       color='color',
       size=5)
