import streamlit as st
import pandas as pd
import random
import time
from custom import custom_functions
import plotly.express as px
import plotly.graph_objects as go

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
st.title("Exploring the raw dataset")

cleaned_df = pd.read_pickle("data/pickles/total_listings_viz.pkl")
raw_df = custom_functions.retrieve_raw_dataset()

delta_rows = cleaned_df.shape[0] - raw_df.shape[0]
delta_columns = cleaned_df.shape[1] - raw_df.shape[1]
delta_NAs = (cleaned_df.shape[0] * cleaned_df.shape[1] - (sum(cleaned_df.count()))) - (
    raw_df.shape[0] * raw_df.shape[1] - (sum(raw_df.count()))
)

st.write("## Overview")
st.markdown(
    """
Description: Provide a brief description of the dataset, including the number of rows, columns, and data types.
"""
)
st.write("## Dataset preview")
if st.toggle("Take a peak to the cleaned dataset"):
    col1, col2, col3 = st.columns(3)
    col1.metric(
        label="Observations",
        value=str(cleaned_df.shape[0]),
        delta=delta_rows,
        delta_color="inverse",
    )
    col2.metric(
        label="Features",
        value=str(cleaned_df.shape[1]),
        delta=delta_columns,
        delta_color="inverse",
    )
    col3.metric(
        label="NAs",
        value=str(
            cleaned_df.shape[0] * cleaned_df.shape[1] - (sum(cleaned_df.count()))
        ),
        help="*NAs cells in the dataframe. These left NAs are imputed in the predictive model pipeline*",
        delta=delta_NAs,
        delta_color="inverse",
    )
    with st.spinner("Wait for clean dataset preview to load..."):
        time.sleep(2)
    st.dataframe(cleaned_df.head())
    st.toast("**Clean dataset is in memory, toggle again to return to raw dataset**")

else:
    with st.spinner("Wait for raw dataset preview to load..."):
        time.sleep(2)
    col1, col2, col3 = st.columns(3)
    col1.metric(
        label="Observations",
        value=str(raw_df.shape[0]),
        delta=-delta_rows,
        delta_color="inverse",
    )
    col2.metric(
        label="Features",
        value=str(raw_df.shape[1]),
        delta=-delta_columns,
        delta_color="inverse",
    )
    col3.metric(
        label="NAs",
        value=raw_df.shape[0] * raw_df.shape[1] - (sum(raw_df.count())),
        help="*NAs cells in the dataframe.*",
        delta=-delta_NAs,
        delta_color="inverse",
    )

    st.dataframe(raw_df.head().style.apply(custom_functions.color_coding, axis=0))

    st.write("## Data types")
    st.markdown(
        """
    Add some text to describe the bar chart showing the distribution of different data types
    """
    )

    fig = px.histogram(
        pd.DataFrame(raw_df.dtypes.astype(str), columns=["types"]),
        x="types",
        color_discrete_sequence=["black"] * len(raw_df.dtypes),
    )

    st.plotly_chart(fig)

    st.write("## Missing values")
    st.write("### Features")
    st.markdown(
        """
    Add some text to describe the bar chart highlighting features with missing values.
    """
    )

    NAs_df = custom_functions.plot_nas_columns(df=raw_df).sort_values(
        "NAs", ascending=False
    )

    fig = px.bar(
        NAs_df,
        x=NAs_df.index,
        y="NAs",
        text_auto=True,
        color="color",
        color_discrete_map={"Remove": "red", "Keep": "black"}
    )

    st.plotly_chart(fig)

    st.write("### Observations")
    st.markdown(
        """
    Add some text to describe the bar chart highlighting observations with missing values
    """
    )
    NAs_df = custom_functions.plot_nas_rows(df=raw_df, n=0)

    fig = px.histogram(
        NAs_df, x="NAs", text_auto=True, color_discrete_sequence=["black"] * len(NAs_df)
    )

    fig.add_vline(x=7.5, line_color="red", line_width=2, line_dash="dash")

    st.plotly_chart(fig)

    st.toast("**All the objects of the page are loaded!**")
