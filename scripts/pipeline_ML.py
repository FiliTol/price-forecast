import pandas as pd
from feature_engine.datetime import DatetimeSubtraction
from feature_engine.creation import RelativeFeatures
from feature_engine.encoding import OneHotEncoder, CountFrequencyEncoder, OrdinalEncoder
from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.imputation import MeanMedianImputer, AddMissingIndicator, CategoricalImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


df = pd.read_pickle("data/pickles/listings_viz_sep.pkl")
review_dates_feature = ["first_review",  # Variable possibly not needed
                        "last_review"]

host_listings_feature = ["host_listings_count",
                         "host_total_listings_count"]

ohe_feature = ["neighbourhood_cleansed",  # categorical
               "host_is_superhost",  # binary
               "host_has_profile_pic",
               "host_identity_verified",
               "email_verification",
               "phone_verification",
               "work_email_verification"]

ordinal_feature = ["host_response_time",
                   "room_type",
                   "bathrooms_text"]

host_id_feature = ["host_id"]

host_since_feature = ["host_since"]

numerical_feature = ["host_response_rate",
                     "host_acceptance_rate",
                     "host_location",
                     "minimum_nights",
                     "maximum_nights",
                     "number_of_reviews",
                     "review_scores_rating",
                     "review_scores_accuracy",
                     "review_scores_cleanliness",
                     "review_scores_checkin",
                     "review_scores_communication",
                     "review_scores_location",
                     "review_scores_value",
                     "reviews_per_month",
                     "airport_distance_km",
                     "ferretto_square_distance_km",
                     "roma_square_distance_km",
                     "rialto_bridge_distance_km",
                     "san_marco_square_distance_km"
                     ]

coordinates_feature = ["latitude",
                       "longitude"]

accomodates_vs_feature = ["accommodates",
                          "bathrooms",
                          "bedrooms",
                          "beds"]

bedrooms_feature = ["beds",
                    "bedrooms"]

calculated_listings_feature = ["calculated_host_listings_count",
                               "calculated_host_listings_count_entire_homes",
                               "calculated_host_listings_count_private_rooms",
                               "calculated_host_listings_count_shared_rooms"
                               ]

## Drop rows with NaN in target 
df = df.loc[df['price'].notnull(), :]

X = df.drop(["price"], axis=1, inplace=False)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=874631)

wizard_pipe = Pipeline(steps=[
    # Review Dates (RD)
    ('RD_engineering', DatetimeSubtraction(variables="last_review",
                                           reference="first_review",
                                           output_unit="D",
                                           drop_original=True,
                                           new_variables_names=["days_active_reviews"],
                                           missing_values="ignore"
                                           )),
    ("RD_imputation", MeanMedianImputer(imputation_method="median",
                                        variables=["days_active_reviews"])),
    # =========================
    # Host Listings Count (HLC)
    ("HLC_imputation", MeanMedianImputer(imputation_method="median",
                                         variables=host_listings_feature)),
    ("HLC_engineering", RelativeFeatures(variables=['host_listings_count'],
                                         reference=['host_total_listings_count'],
                                         func=['div'],
                                         drop_original=True)),
    # ========================
    # One-hot-encoding (OHE)
    ("OHE_imputation", CategoricalImputer(imputation_method="frequent",
                                          variables=ohe_feature,
                                          return_object=True,
                                          ignore_format=False)),
    ("OHE_encoding", OneHotEncoder(top_categories=None,
                                   drop_last=True,
                                   drop_last_binary=True,
                                   ignore_format=False,
                                   variables=ohe_feature)),
    # =======================
    # Ordinal encoding (OE)
    ("OE_imputation", CategoricalImputer(imputation_method="frequent",
                                         variables=ordinal_feature,
                                         return_object=True,
                                         ignore_format=False)),
    ("OE_encoding", OrdinalEncoder(encoding_method="ordered",
                                   variables=ordinal_feature,
                                   missing_values="ignore",
                                   ignore_format=False,
                                   unseen="encode")),
    # ========================
    # Host ID (HID)
    ("HID_imputation", CategoricalImputer(imputation_method="missing",
                                          variables=host_id_feature,
                                          fill_value="MISSING")),
    ("HID_encoding", CountFrequencyEncoder(encoding_method="count",
                                           missing_values="ignore",
                                           unseen="encode")),
    # =========================
    # Host since (HS)
    ("HS_engineering", DatetimeSubtraction(variables=max("last_review"),
                                           reference="host_since",
                                           output_unit="D",
                                           drop_original=True,
                                           new_variables_names=["host_since_days"],
                                           missing_values="ignore")),
    ("HS_imputation", MeanMedianImputer(imputation_method="median",
                                        variables=host_since_feature)),
    # ==========================
    # Numerical features (NF)
    ("NF_imputation", SklearnTransformerWrapper(transformer=KNNImputer(n_neighbors=3,
                                                                       weights="uniform"),
                                                variables=numerical_feature)),
    # ============================
    # Coordinates numerical (COO)
    ("COO_imputation", MeanMedianImputer(imputation_method="median",
                                         variables=coordinates_feature)),
    # ============================
    # Accomodates VS (AVS)
    ("AVS_engineering", RelativeFeatures(variables=["bathrooms",
                                                    "bedrooms",
                                                    "beds"],
                                         reference=["accommodates"],
                                         func=["div"],
                                         fill_value=None,
                                         missing_values="ignore",
                                         drop_original=True
                                         )),
    ("AVS_imputation", MeanMedianImputer(imputation_method="median",
                                         variables=["bathrooms_div_accommodates",
                                                    "bedrooms_div_accommodates",
                                                    "beds_div_accommodates"])),
    # =======================
    # Beds VS Rooms (BVR)
    ("BVR_engineering", RelativeFeatures(variables=["beds"],
                                         reference=["bedrooms"],
                                         func=["div"],
                                         fill_value=None,
                                         missing_values="ignore",
                                         drop_original=True)),
    ("BVR_imputation", MeanMedianImputer(imputation_method="median",
                                         variables=["beds_div_bedrooms"])),
    # =========================
    # Calculated Host Listings (CHL)
    ("CHL_engineering", RelativeFeatures(variables=["calculated_host_listings_count_entire_homes",
                                                    "calculated_host_listings_count_private_rooms",
                                                    "calculated_host_listings_count_shared_rooms"],
                                         reference=["calculated_host_listings_count"],
                                         func=["div"],
                                         fill_value=None,
                                         missing_values="ignore",
                                         drop_original=True)),
    ("CHL_imputation", MeanMedianImputer(imputation_method="median",
                                         variables=["calculated_host_listings_count_entire_homes_div_calculated_host_listings_count",
                                                    "calculated_host_listings_count_private_rooms_div_calculated_host_listings_count",
                                                    "calculated_host_listings_count_shared_rooms_div_calculated_host_listings_count"])),

    # ======================================================================
    # Scaling
    # ======================================================================
    ("MinMaxScaling", SklearnTransformerWrapper(transformer=MinMaxScaler(),
                                                variables=["days_active_reviews",
                                                           "host_listings_count_div_host_total_listings_count",
                                                           "host_since_days",
                                                           "bathrooms_div_accommodates",
                                                           "bedrooms_div_accommodates",
                                                           "beds_div_accommodates",
                                                           "beds_div_bedrooms",
                                                           "calculated_host_listings_count_entire_homes_div_calculated_host_listings_count",
                                                           "calculated_host_listings_count_private_rooms_div_calculated_host_listings_count",
                                                           "calculated_host_listings_count_shared_rooms_div_calculated_host_listings_count"
                                                           ] + numerical_feature)),

    ("StandardScaler", SklearnTransformerWrapper(transformer=StandardScaler(),
                                                 variables=coordinates_feature)),

    # ============
    # Prediction
    # ============
    ("RandomForestRegressor", RandomForestRegressor())

]
)


model = wizard_pipe.fit(X_train,y_train)












