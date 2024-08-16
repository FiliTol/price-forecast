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


def color_coding(col):
    removed_features = ["neighborhood_overview",
                        "host_about",
                         "host_neighbourhood",
                         "neighbourhood",
                         "neighbourhood_group_cleansed",
                         "calendar_updated",
                         "license",
                         "listing_url",
                         "scrape_id",
                         "last_scraped",
                         "source",
                         "name",
                         "description",
                         "picture_url",
                         "host_url",
                         "host_name",
                         "host_thumbnail_url",
                         "host_picture_url",
                         "minimum_minimum_nights",
                         "maximum_minimum_nights",
                         "minimum_maximum_nights",
                         "maximum_maximum_nights",
                         "minimum_nights_avg_ntm",
                         "maximum_nights_avg_ntm",
                         "has_availability",
                         "availability_30",
                         "availability_60",
                         "availability_90",
                         "availability_365",
                         "calendar_last_scraped",
                         "number_of_reviews_ltm",
                         "number_of_reviews_l30d",
                         "instant_bookable",
                         "calculated_host_listings_count",
                         "calculated_host_listings_count_entire_homes",
                         "calculated_host_listings_count_private_rooms",
                         "calculated_host_listings_count_shared_rooms"
                        ]

    return ['color:white;background-color:red'] * len(
        col) if col.name in removed_features else ['background-color:green'] * len(col)


def plot_nas_columns(df: pd.DataFrame):
    deleted_features = ["neighborhood_overview",
                        "host_about",
                        "host_neighbourhood",
                        "neighbourhood",
                        "neighbourhood_group_cleansed",
                        "calendar_updated",
                        "license"]
    df_nas = pd.DataFrame(df.isnull().sum(), columns=["NAs"])
    df_nas.reset_index(inplace=True)
    df_nas['color'] = df_nas['index'].apply(lambda x: 'Remove' if x in deleted_features else 'Keep')
    df_nas.set_index("index", inplace=True)
    return df_nas.loc[df_nas["NAs"] > 0, :]


def plot_nas_rows(df: pd.DataFrame, n: int):
    df_nas_rows = pd.DataFrame({
        'NAs': df.isnull().sum(axis=1),
        'Columns_with_NAs': df.apply(lambda x: ', '.join(x.index[x.isnull()]), axis=1)
    })
    return df_nas_rows.loc[df_nas_rows["NAs"] > n]
