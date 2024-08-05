import pandas as pd
import re
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scripts.custom.tools import JsonHandler
from scripts.custom.viz.class_transformers import GeographicTransformer, CreateStrategicLocationTransformer, \
    VectorToDataFrame, NeighborhoodMapper, BathroomsTransformer, CreateVerificationsTransformer
from scripts.custom.viz.function_transformers import fun_tr_transform_nan_unicode, fun_tr_id_to_string, \
    fun_tr_from_string_to_rate, fun_tr_transform_to_datetime, fun_tr_remove_dollar_sign
from sklearn.utils import estimator_html_repr
from sklearn import set_config
import sys

set_config(transform_output="pandas")


pd.options.display.float_format = "{:.0f}".format
handler = JsonHandler()

host_locations = handler.import_from_json("data/mappings/host_locations.json")
strategic_locations = handler.import_from_json("data/mappings/strategic_locations.json")
neighbourhood_levels = handler.import_from_json(
    "data/mappings/neighbourhoods_levels.json"
)
remap_baths = handler.import_from_json("data/mappings/baths.json")

period = sys.argv[1]

df_listings = pd.read_csv(f"data/data_{period}/d_listings.csv")
df_listings.drop(
    labels=[
        "listing_url",
        "name",
        "scrape_id",
        "last_scraped",
        "source",
        "description",
        "picture_url",
        "host_url",
        "host_name",
        "host_thumbnail_url",
        "host_picture_url",
        "host_neighbourhood",
        "neighbourhood",
        "neighbourhood_group_cleansed",
        "property_type",
        "amenities",
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
        "calendar_updated",
        "calendar_last_scraped",
        "number_of_reviews_ltm",
        "number_of_reviews_l30d",
        "license",
        "instant_bookable",
    ],
    axis=1,
    inplace=True,
)

string_features = ["neighborhood_overview", "host_about"]
id_feature = ["id", "host_id"]
rate_feature = ["host_response_rate", "host_acceptance_rate"]
time_feature = ["host_since", "first_review", "last_review"]
neighbourhood_feature = ["neighbourhood_cleansed"]
price_feature = ["price"]

text_encoding_pipeline = Pipeline(
    steps=[
        ("text preprocessing", fun_tr_transform_nan_unicode),
        (
            "tf-idf vectorizer",
            TfidfVectorizer(
                encoding="utf-8",
                decode_error="ignore",
                strip_accents="unicode",
                lowercase=True,
                analyzer="word",
                max_df=0.8,
                use_idf=True,
                smooth_idf=True,
                max_features=30,
            ),
        ),
        ("Vectors into dataframe", VectorToDataFrame()),
    ]
)

id_pipeline = Pipeline(steps=[("From ID to string", fun_tr_id_to_string)])

rates_pipeline = Pipeline(
    steps=[("Transform response rate", fun_tr_from_string_to_rate)]
)

timestamp_pipeline = Pipeline(
    steps=[("Transform to timestamp", fun_tr_transform_to_datetime)]
)

neighbourhood_pipeline = Pipeline(
    steps=[("Neighbourhood Mapper", NeighborhoodMapper(mapping=neighbourhood_levels))]
)

price_pipeline = Pipeline(steps=[("Trim price feature", fun_tr_remove_dollar_sign)])

# Apply to all dataset (feature engineering using other features)
feature_creation_pipeline = Pipeline(
    steps=[
        (
            "Strategic locations distance",
            CreateStrategicLocationTransformer(locations=strategic_locations),
        ),
        (
            "Host location",
            GeographicTransformer(column="host_location", locations=host_locations),
        ),
        ("Host verifications", CreateVerificationsTransformer()),
        ("Bathrooms", BathroomsTransformer(remap_baths)),
    ]
)

df_listings = feature_creation_pipeline.fit_transform(df_listings)

feature_preprocessor = ColumnTransformer(
    remainder="passthrough",
    n_jobs=-1,
    force_int_remainder_cols=False,
    transformers=[
        # ("Text encoding", text_encoding_pipeline, string_features),
        ("Id", id_pipeline, id_feature),
        ("Rates", rates_pipeline, rate_feature),
        ("Neighbourhood", neighbourhood_pipeline, neighbourhood_feature),
        ("Price", price_pipeline, price_feature),
        ("Timestamp", timestamp_pipeline, time_feature),
    ],
)

cleaned_df = feature_preprocessor.fit_transform(df_listings)


def remove_before_double_underscore(input_string):
    result = re.sub(r'^.*?__', '', input_string)
    return result


def return_cleaned_col_names(list_of_names: list) -> list:
    cleaned_names = []
    for name in list_of_names:
        cleaned_names.append(remove_before_double_underscore(name))
    return cleaned_names


cleaned_df.columns = return_cleaned_col_names(cleaned_df.columns)

pd.to_pickle(cleaned_df,
             f"data/pickles/listings_viz_{period}.pkl",
             )

with open("data/visual/feature_preprocessor.html", "w") as f:
    f.write(estimator_html_repr(feature_preprocessor))
