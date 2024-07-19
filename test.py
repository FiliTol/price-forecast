import streamlit as st
import pandas as pd
import random

random.seed(874631)


def random_color_generator():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return r, g, b


random_color = random_color_generator()

df = pd.read_csv("data/listings.csv")
graph = df[['id', 'host_id', 'latitude', 'longitude', 'host_neighbourhood']]

zona = st.sidebar.selectbox(
    "Which sestiere do you want to see?",
    graph['host_neighbourhood'].unique().tolist()
)

col_zona = {}
for i in graph['host_neighbourhood'].unique():
    col_zona[i] = random_color_generator()

graph['color'] = graph['host_neighbourhood'].apply(lambda x: col_zona.get(x))

st.map(graph.loc[graph['host_neighbourhood'] == zona],
       color='color',
       size=5)

st.write("# My first dashboard")
