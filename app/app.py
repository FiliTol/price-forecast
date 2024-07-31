import streamlit as st
import pandas as pd
import random
from custom.custom_functions import random_color_generator

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
st.title("Listings overview")

df = pd.read_pickle("data/pickles/december_listings_viz.pkl")
df.fillna("MISSING", inplace=True)
df['host_response_rate'] = pd.to_numeric(df['host_response_rate'], errors='coerce')
df['host_acceptance_rate'] = pd.to_numeric(df['host_acceptance_rate'], errors='coerce')
df['price'] = pd.to_numeric(df['price'], errors="coerce")
df['host_since'] = pd.to_datetime(df['host_since'], errors="coerce").dt.date
df['host_location'] = pd.to_numeric(df['host_location'], errors="coerce")
graph = df.copy()

zona_options = sorted(graph.loc[:, 'neighbourhood_cleansed'].unique().tolist())
zona = st.sidebar.multiselect('Select neighbourhoods:', zona_options + ["All"], default="All")
if "All" in zona:
    zona = zona_options

response_time_list = sorted(graph['host_response_time'].unique().tolist())
response_time = st.sidebar.multiselect('Select response time:', response_time_list + ["All"], default="All")
if "All" in response_time:
    response_time = response_time_list

response_rate = st.sidebar.slider("Select host response rate", min_value=0, max_value=100, value=100)
if graph['host_response_rate'].isnull().sum()>0:
    response_rate_MISSING_box = st.sidebar.toggle("Include NA in visualization for Host Response Rate", value=False)

acceptance_rate = st.sidebar.slider("Select host acceptance rate", min_value=0, max_value=100, value=100)
if graph['host_acceptance_rate'].isnull().sum()>0:
    acceptance_rate_MISSING_box = st.sidebar.toggle("Include NA in visualization for Host Acceptance Rate", value=False)

price_range = st.sidebar.slider("Select price range",
                                min_value=0.0,
                                max_value=max(graph['price']),
                                value=(0.0, max(graph['price'])),
                                step=0.01)
if graph['price'].isnull().sum()>0:
    price_range_MISSING_box = st.sidebar.toggle("Include NA in visualization for Price", value=False)

host_since = st.sidebar.date_input("Host since", min(graph['host_since']))

host_distance = st.sidebar.slider("Distance between host house and listing location (0 km to 100+ km)",
                                  min_value=min(graph['host_location']),
                                  max_value=100.0,
                                  value=(0.0, 20.0),
                                  step=0.01)
if host_distance == 100.0:
    host_distance = max(graph['host_location'])
if graph['host_location'].isnull().sum()>0:
    host_distance_MISSING_box = st.sidebar.toggle("Include NA in visualization for host distance", value=False)

superhost_list = sorted(graph['host_is_superhost'].unique().tolist())
superhost = st.sidebar.multiselect('Is host a superhost?:', superhost_list + ["All"], default="All")
if "All" in superhost:
    superhost = superhost_list

is_in_neighbourhood = graph['neighbourhood_cleansed'].isin(zona)
is_in_response_time = graph['host_response_time'].isin(response_time)
is_response_rate_valid = (graph['host_response_rate'] <= response_rate) | response_rate_MISSING_box
is_acceptance_rate_valid = (graph['host_acceptance_rate'] <= acceptance_rate) | acceptance_rate_MISSING_box
is_in_price_range = (price_range[0] < graph['price']) & (graph['price'] < price_range[1]) | price_range_MISSING_box
is_host_since = graph['host_since'] >= host_since
is_host_distance = (host_distance[0] < graph['host_location']) & (graph['host_location'] < host_distance[1]) | host_distance_MISSING_box
is_host_superhost = graph['host_is_superhost'].isin(superhost)

filtered_graph = graph[
    is_in_neighbourhood & 
    is_in_response_time & 
    is_response_rate_valid & 
    is_acceptance_rate_valid &
    is_in_price_range &
    is_host_since &
    is_host_distance &
    is_host_superhost
]

st.metric("Selected listings:", value=f"{filtered_graph.shape[0]}/{graph.shape[0]}")
st.map(filtered_graph,
       size=5,
       use_container_width=False)
