import pandas as pd
import random
import time
import os
import re
from pandarallel import pandarallel


def random_color_generator():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return r, g, b


def stream_text(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.1)


def retrieve_raw_dataset():
    datasets = {}
    for file in os.listdir("data/all_cities"):
        pattern = r'_(\w{2})'
        match = re.search(pattern, file)
        result = match.group(1)
        datasets[f"df_{result}"] = pd.read_csv(f"data/all_cities/{file}")
    return pd.concat([value for key, value in datasets.items()], ignore_index=True)


def plot_nas_columns(df: pd.DataFrame):
    df_nas = pd.DataFrame(df.isnull().sum(), columns=["NAs"])
    return df_nas.loc[df_nas["NAs"] > 0, :]


def plot_nas_rows(df: pd.DataFrame, n: int):
    df_nas_rows = pd.DataFrame({
        'NAs': df.isnull().sum(axis=1),
        'Columns_with_NAs': df.apply(lambda x: ', '.join(x.index[x.isnull()]), axis=1)
    })
    return df_nas_rows.loc[df_nas_rows["NAs"] > n]
