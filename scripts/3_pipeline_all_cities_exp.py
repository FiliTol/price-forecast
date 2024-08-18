import pandas as pd
from sklearn.pipeline import Pipeline
from custom.class_transformers import ColumnDropperTransformer, IntoBinaryTransformer
from sklearn import set_config

set_config(transform_output="pandas")
pd.options.display.float_format = "{:.0f}".format

df = pd.read_pickle("data/pickles/total_listings_viz.pkl")

# Columns to drop because of high correlation
to_drop_corr = [
    "host_acceptance_rate",
    "host_total_listings_count",
    "bathrooms",
    "bedrooms",
    "beds",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "amenities_other",
    "amenities_toiletry",
    "amenities",
]

widely_unbalanced_features = [
    "host_has_profile_pic",
    "host_identity_verified",
    "email_verification",
    "phone_verification",
    "work_email_verification",
]

eng_after_exploration_pipeline = Pipeline(
    steps=[
        ("Drop NAs columns", ColumnDropperTransformer(columns=to_drop_corr)),
        (
            "Drop unbalanced columns",
            ColumnDropperTransformer(columns=widely_unbalanced_features),
        ),
        (
            "Transform Response Rate",
            IntoBinaryTransformer(
                feature="host_response_rate", cat1="100", cond="x==100", cat2="lower"
            ),
        ),
        (
            "Transform Minimum Nights",
            IntoBinaryTransformer(
                feature="minimum_nights", cat1="1", cond="x<=1", cat2="more_than_1"
            ),
        ),
        (
            "Transform Maximum Nights",
            IntoBinaryTransformer(
                feature="maximum_nights",
                cat1="less_than_100",
                cond="x<=100",
                cat2="more_than_100",
            ),
        ),
        (
            "Transform City Population",
            IntoBinaryTransformer(
                feature="listing_city_pop",
                cat1="less_than_300k",
                cond="x<=300000",
                cat2="more_than_300k",
            ),
        ),
        (
            "Transform Review Score Rating",
            IntoBinaryTransformer(
                feature="review_scores_rating",
                cat1="less_than_4.8",
                cond="x<4.8",
                cat2="more_than_4.8",
            ),
        ),
        (
            "Transform Host Respnse Time",
            IntoBinaryTransformer(
                feature="host_response_time",
                cat1="within_an_hour",
                cond="x=='within an hour'",
                cat2="more_than_one_hour",
            ),
        ),
        (
            "Transform Property Type",
            IntoBinaryTransformer(
                feature="property_type",
                cat1="entire_property",
                cond="x=='entire_property'",
                cat2="other",
            ),
        ),
        (
            "Transform Room Type",
            IntoBinaryTransformer(
                feature="room_type",
                cat1="entire_home",
                cond="x=='Entire home/apt'",
                cat2="other",
            ),
        ),
        (
            "Transform Bathrooms Text",
            IntoBinaryTransformer(
                feature="bathrooms_text",
                cat1="single",
                cond="x=='single'",
                cat2="other",
            ),
        ),
    ],
    verbose=True,
)

df = eng_after_exploration_pipeline.fit_transform(df)
pd.to_pickle(df, "data/pickles/total_listings_exploration_handling.pkl")
