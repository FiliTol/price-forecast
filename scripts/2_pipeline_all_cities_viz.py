import pandas as pd
from pandarallel import pandarallel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.tools import JsonHandler, concatenate_listings_datasets, return_cleaned_col_names
from src.class_transformers import (
    GeographicTransformer,
    BathroomsTransformer,
    CreateVerificationsTransformer,
    AmenitiesTransformer,
    OfflineLocationFinder,
    PropertyTypeTransformer,
    HostLocationImputer,
)
from src.function_transformers import (
    fun_tr_id_to_string,
    fun_tr_from_string_to_rate,
    fun_tr_transform_to_datetime,
    fun_tr_remove_dollar_sign,
)
from sklearn.utils import estimator_html_repr
from sklearn import set_config

set_config(transform_output="pandas")

pandarallel.initialize()
pd.options.display.float_format = "{:.0f}".format
handler = JsonHandler()

print("Importing dataset and other data...")
df_listings = concatenate_listings_datasets()
host_locations = handler.import_from_json("data/mappings/host_locations.json")
remap_baths = handler.import_from_json("data/mappings/baths.json")
print("Data imported!")

print("Dropping columns and rows with too many NAs...")
df_listings.drop(
    [
        "neighborhood_overview",
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
        "calculated_host_listings_count_shared_rooms",
    ],
    axis=1,
    inplace=True,
)

df_listings.set_index("id", inplace=True)

df_nas_columns = pd.DataFrame(
    {
        "NAs": df_listings.isnull().sum(axis=1),
        "Columns_with_NAs": df_listings.apply(
            lambda x: ", ".join(x.index[x.isnull()]), axis=1
        ),
    }
)

more_than_7_missing = df_nas_columns.loc[df_nas_columns["NAs"] > 7, :].index.tolist()
df_listings.drop(more_than_7_missing, inplace=True)
print("Columns and rows dropping completed!")


id_feature = ["host_id"]
rate_feature = ["host_response_rate", "host_acceptance_rate"]
time_feature = ["host_since", "first_review", "last_review"]
neighbourhood_feature = ["neighbourhood_cleansed"]
price_feature = ["price"]

# Amenities
technology_pattern: str = (
    r"\b(wifi|internet|ethernet|cable|fibra|dolby|smart|connection|tv|television|netflix|amazon|disney)\b"
)
kitchen_pattern: str = (
    r"\b(kitchen|cooking|grill|cucina|refrigerator|fridge|oven|stove|dish|coffee|espresso|lavazza|dining|breakfast|microonde|microwave|washer|freezer|glasses|toast|baking)\b"
)
toiletry_pattern: str = (
    r"\b(hair|capelli|soap|sapone|bidet|shampoo|bathtub|gel|laundry|closet|pillow|blanket|shower)\b"
)
acheating_pattern: str = r"\b(heating|ac|air|conditioning|fan)\b"
benefits_pattern: str = (
    r"\b(garden|backyard|skyline|beach|gym|fitness|view|outdoor|balcony|waterfront|bed linen|workspace|aid|luggage|elevator|free|safe|lock|security|bike|estinguisher)\b"
)
other_amenity_pattern: str = (
    r"\b(wifi|internet|ethernet|cable|fibra|dolby|smart|connection|tv|television|netflix|amazon|disney|kitchen|cooking|grill|cucina|refrigerator|fridge|oven|stove|dish|coffee|espresso|lavazza|dining|breakfast|microonde|microwave|washer|freezer|glasses|toast|baking|hair|capelli|soap|sapone|bidet|shampoo|bathtub|gel|laundry|closet|pillow|blanket|shower|heating|ac|air|conditioning|fan|garden|backyard|skyline|beach|gym|fitness|view|outdoor|balcony|waterfront|bed linen|workspace|aid|luggage|elevator|free|safe|lock|security|bike|estinguisher)\b"
)

set_amenities_remapper = [
    (technology_pattern, "technology"),
    (kitchen_pattern, "kitchen"),
    (toiletry_pattern, "toiletry"),
    (acheating_pattern, "AC/heating"),
    (benefits_pattern, "benefits"),
    (other_amenity_pattern, "other"),
]

# Property type
entire_property_pattern = r"\b(entire|tiny home)\b"
private_room_pattern = r"\b(private room|room in serviced apartment|room in bed and breakfast|room in hotel|room in resort)\b"
shared_room_pattern = r"\b(shared room|shared)\b"
other_room_pattern = r"\b(entire|tiny home|private room|room in serviced apartment|room in bed and breakfast|room in hotel|room in resort|shared room|shared)\b"

set_property_type_remapper = [
    (entire_property_pattern, "entire_property"),
    (private_room_pattern, "private_room"),
    (shared_room_pattern, "shared_room"),
    (other_room_pattern, "other"),
]

id_pipeline = Pipeline(steps=[("From ID to string", fun_tr_id_to_string)], verbose=True)

rates_pipeline = Pipeline(
    steps=[("Transform response rate", fun_tr_from_string_to_rate)], verbose=True
)

timestamp_pipeline = Pipeline(
    steps=[("Transform to timestamp", fun_tr_transform_to_datetime)], verbose=True
)

price_pipeline = Pipeline(
    steps=[("Trim price feature", fun_tr_remove_dollar_sign)], verbose=True
)

# Apply to all dataset (feature engineering using other features)
feature_creation_pipeline = Pipeline(
    steps=[
        ("Listing Locations", OfflineLocationFinder()),
        ("Host Locations imputer", HostLocationImputer()),
        (
            "Host location",
            GeographicTransformer(column="host_location", locations=host_locations),
        ),
        ("Host verifications", CreateVerificationsTransformer()),
        ("Bathrooms", BathroomsTransformer(remap_baths)),
        (
            "Amenities",
            AmenitiesTransformer(df=df_listings, remapper=set_amenities_remapper),
        ),
        (
            "Property type",
            PropertyTypeTransformer(
                df=df_listings, remapper=set_property_type_remapper
            ),
        ),
    ],
    verbose=True,
)

print("Executing Feature Creation Pipeline...")
df_listings = feature_creation_pipeline.fit_transform(df_listings)
print("Feature Creation Pipeline completed!")

print("Executing preprocessing on features...")
feature_preprocessor = ColumnTransformer(
    remainder="passthrough",
    n_jobs=-1,
    force_int_remainder_cols=False,
    transformers=[
        ("Id", id_pipeline, id_feature),
        ("Rates", rates_pipeline, rate_feature),
        ("Price", price_pipeline, price_feature),
        ("Timestamp", timestamp_pipeline, time_feature),
    ],
    verbose=True,
)

cleaned_df = feature_preprocessor.fit_transform(df_listings)
print("Preprocessing on features completed!")

cleaned_df.columns = return_cleaned_col_names(cleaned_df.columns)

pd.to_pickle(
    cleaned_df,
    f"data/pickles/total_listings_viz.pkl",
)

with open("data/visual/feature_preprocessor.html", "w") as f:
    f.write(estimator_html_repr(feature_preprocessor))
print("Viz pipeline execution completed")
