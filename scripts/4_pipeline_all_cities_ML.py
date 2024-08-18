import pandas as pd
from feature_engine.datetime import DatetimeSubtraction
from feature_engine.creation import RelativeFeatures
from feature_engine.encoding import OneHotEncoder, CountFrequencyEncoder, OrdinalEncoder
from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
import sys


df = pd.read_pickle("data/pickles/total_listings_exploration_handling.pkl")

review_dates_feature = ["first_review", "last_review"]

ohe_feature = [
    "host_is_superhost",
    "host_response_rate",
    "host_response_time",
    "minimum_nights",
    "maximum_nights",
    "listing_city_pop",
    "review_scores_rating",
    "property_type",
    "room_type",
    "bathrooms_text",
]

ohe_most_frequent = ["listing_city", "neighbourhood_cleansed"]

host_id_feature = ["host_id"]

host_since_feature = ["host_since"]

numerical_feature = [
    "host_listings_count",
    "host_location",
    "number_of_reviews",
    "reviews_per_month",
    "amenities_AC/heating",
    "amenities_technology",
    "amenities_kitchen",
    "amenities_benefits",
    "accommodates",
]

coordinates_feature = ["latitude", "longitude"]

# Add feature needed for feature engineering of host_since
df["scraping_date"] = max(df["last_review"])

# Drop rows with NaN in target
df = df.loc[df["price"].notnull(), :]

X = df.drop(["price"], axis=1, inplace=False)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=874631
)

wizard_pipe = Pipeline(
    steps=[
        # Review Dates (RD)
        (
            "RD_engineering",
            DatetimeSubtraction(
                variables="last_review",
                reference="first_review",
                output_unit="D",
                drop_original=True,
                new_variables_names=["days_active_reviews"],
                missing_values="ignore",
            ),
        ),
        (
            "RD_imputation",
            MeanMedianImputer(
                imputation_method="median", variables=["days_active_reviews"]
            ),
        ),
        # ========================
        # One-hot-encoding (OHE)
        (
            "OHE_imputation",
            CategoricalImputer(
                imputation_method="frequent",
                variables=ohe_feature,
                return_object=True,
                ignore_format=False,
            ),
        ),
        (
            "OHE_encoding",
            OneHotEncoder(
                top_categories=None,
                drop_last=True,
                drop_last_binary=True,
                ignore_format=False,
                variables=ohe_feature,
            ),
        ),
        # ========================
        # One-hot-encoding Top Frequent (OHETF)
        (
            "OHETF_imputation",
            CategoricalImputer(
                imputation_method="frequent",
                variables=ohe_most_frequent,
                return_object=True,
                ignore_format=False,
            ),
        ),
        (
            "OHETF_encoding",
            OneHotEncoder(
                top_categories=5,
                drop_last=True,
                drop_last_binary=True,
                ignore_format=False,
                variables=ohe_most_frequent,
            ),
        ),
        # =======================
        # Host ID (HID)
        (
            "HID_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=host_id_feature,
                fill_value="MISSING",
            ),
        ),
        (
            "HID_encoding",
            CountFrequencyEncoder(
                encoding_method="count", missing_values="ignore", unseen="encode"
            ),
        ),
        # =========================
        # Host since (HS)
        (
            "HS_engineering",
            DatetimeSubtraction(
                variables=["scraping_date"],
                reference=["host_since"],
                output_unit="D",
                drop_original=True,
                new_variables_names=["host_since_days"],
                missing_values="ignore",
            ),
        ),
        (
            "HS_imputation",
            MeanMedianImputer(
                imputation_method="median", variables=["host_since_days"]
            ),
        ),
        # ==========================
        # Numerical features (NF)
        (
            "NF_imputation",
            SklearnTransformerWrapper(
                transformer=KNNImputer(n_neighbors=5, weights="uniform"),
                variables=numerical_feature,
            ),
        ),
        # ============================
        # Coordinates numerical (COO)
        (
            "COO_imputation",
            MeanMedianImputer(
                imputation_method="median", variables=coordinates_feature
            ),
        ),
        # =======================
        # Scaling
        # ======================================================================
        (
            "MinMaxScaling",
            SklearnTransformerWrapper(
                transformer=MinMaxScaler(),
                variables=[
                    "days_active_reviews",
                    "host_since_days",
                ]
                + numerical_feature,
            ),
        ),
        (
            "StandardScaler",
            SklearnTransformerWrapper(
                transformer=StandardScaler(), variables=coordinates_feature
            ),
        ),
        # ============
        # Prediction
        # ============
        (
            "RandomForestRegressor",
            RandomForestRegressor(
                n_estimators=100,
                criterion="squared_error",
                bootstrap=True,
                max_samples=0.7,
                oob_score=True,
                n_jobs=-1,
                random_state=874631,
            ),
        ),
        # (
        #    "KNeighborsRegressor",
        #    KNeighborsRegressor(
        #        n_neighbors=5,
        #        weights="uniform",
        #        algorithm="auto",
        #        n_jobs=-1,
        #    )
        # )
    ],
    verbose=True,
)

fitting_model = wizard_pipe.fit(X_train, y_train)
print(
    f"Coefficient of determination R^2 for test set is {wizard_pipe.score(X_test, y_test)}"
)
