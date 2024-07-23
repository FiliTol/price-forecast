import streamlit as st
import pandas as pd
import random

random.seed(874631)

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

df = pd.read_csv("data/listings.csv")
graph = df[['id', 'host_id', 'latitude', 'longitude', 'host_neighbourhood', 'host_response_time']]

zona = st.sidebar.selectbox(
    "Which sestiere do you want to see?",
    graph['host_neighbourhood'].unique().tolist()
)

response_time = st.sidebar.radio('Host answers...',
                                 options=df["host_response_time"].unique().tolist(),
                                 horizontal=True)

col_zona = {}
for i in graph['host_neighbourhood'].unique():
    col_zona[i] = random_color_generator()

graph['color'] = graph['host_neighbourhood'].apply(lambda x: col_zona.get(x))

st.map(graph[(graph['host_neighbourhood'] == zona) & (graph['host_response_time'] == response_time)],
       color='color',
       size=5)
