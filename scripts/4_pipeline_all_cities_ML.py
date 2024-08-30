import pandas as pd
import numpy as np
from feature_engine.datetime import DatetimeSubtraction
from feature_engine.creation import RelativeFeatures
from feature_engine.encoding import OneHotEncoder, CountFrequencyEncoder, OrdinalEncoder
from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures, PowerTransformer, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV
import sys
from sklearn.neural_network import MLPRegressor

param_grid = {
    'regressor__hidden_layer_sizes': [(50,), (100,)],  #, (100, 50), (150, 100, 50)],
    #'regressor__activation': ['relu', 'tanh', 'logistic'],
    'regressor__activation': ["logistic"],
    #'regressor__solver': ['sgd', 'adam'],
    #'regressor__alpha': [0.0001, 0.001, 0.01, 0.1],
    #'regressor__batch_size': ["auto"],
    #'regressor__learning_rate': ['constant', 'invscaling', 'adaptive'],
    #'regressor__learning_rate_init': [0.001, 0.01, 0.1],
    #'regressor__shuffle': [True],
    #'regressor__random_state':[874631],
    #'regressor__tol':[1e-4],
    #'regressor__verbose':[True],
    #'regressor__momentum': [0.8, 0.85, 0.9, 0.95, 0.99],
    #'regressor__n_iter_no_change': [10, 20, 30],
    #'regressor__max_iter': [200, 300, 400],
    #'regressor__warm_start': [True, False],
    #'regressor__early_stopping': [True],
    #'regressor__validation_fraction': [0.1, 0.2, 0.3],
    'transformer__method': ['yeo-johnson'],
    'transformer__standardize': [True],
    'transformer__copy': [False]
}

scoring = {
    'explained_variance': make_scorer(explained_variance_score),
    'mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False),
    'mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
    'r2_score': make_scorer(r2_score),
}

df = pd.read_pickle("data/pickles/total_listings_exploration_handling.pkl")

review_dates_feature = ["first_review", "last_review"]

ohe_feature = [
    "df_city_location",
    "host_is_superhost",
    "host_response_time",
    "property_type",
    "room_type",
    "bathrooms_text",
    "host_response_rate",
    "minimum_nights",
    "maximum_nights",
    "listing_city_pop",
    "review_scores_rating",
    'amenities_internet',
    'amenities_self-checkin',
    'amenities_host-greeting',
    'amenities_pool',
    'amenities_oven',
    'amenities_microwave',
    'amenities_garden',
    'amenities_streaming',
    'amenities_gym',
    'amenities_elevator',
    'amenities_heating',
    "amenities_air-conditioning",
    "amenities_workspace",
    "amenities_freezer",
    "amenities_first-aid-kit",
    "amenities_dishwasher",
    "amenities_long-term-stays",
    "amenities_pets-allowed",
    "amenities_bathtube",
    "amenities_bbq-grill",
    "amenities_lake-bay-view",
]

ohe_most_frequent = ["listing_city", "neighbourhood_cleansed"]

host_id_feature = ["host_id"]

host_since_feature = ["host_since"]

numerical_feature = [
                        "host_listings_count",
                        "host_location",
                        "number_of_reviews",
                        "reviews_per_month",
                        "accommodates",
                    ] #+ description_features

coordinates_feature = [
    "x_coord",
    "y_coord",
    "z_coord"
]

# Add feature needed for feature engineering of host_since
df["scraping_date"] = max(df["last_review"])

# Drop rows with NaN in target
df = df.loc[df["price"].notnull(), :]

X = df.drop(["price"], axis=1, inplace=False)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=874631
)

# Drop the rows from the train set with outliers in price
mask = y_train <= 10000
X_train: np.array = X_train[mask]
y_train: np.array = y_train[mask]

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
                top_categories=7,
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
        #(
        #    "MinMaxScaling",
        #    SklearnTransformerWrapper(
        #        transformer=MinMaxScaler(),
        #        variables=[
        #            "days_active_reviews",
        #            "host_since_days",
        #        ]
        #        + numerical_feature,
        #    ),
        #),
        (
            "PowerTransformer",
            SklearnTransformerWrapper(
                transformer=PowerTransformer(
                    method="yeo-johnson",
                    standardize=True,
                    copy=False
                ),
                variables=[
                              "days_active_reviews",
                              "host_since_days",
                          ]
                          + numerical_feature
            )
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
        #(
        #    "SupportVectorRegression",
        #    SVR(
        #        kernel="rbf",
        #        gamma="auto",
        #        tol=1e-3,
        #        epsilon=0.1,
        #        verbose=True
        #    ),
        #),
        #(
        #    "TransformedTarget-RandomForestRegressor",
        #    TransformedTargetRegressor(regressor=RandomForestRegressor(
        #        n_estimators=100,
        #        criterion="squared_error",
        #        bootstrap=True,
        #        max_samples=0.7,
        #        oob_score=True,
        #        n_jobs=-1,
        #        random_state=874631,
        #    ),
        #        func=np.log,
        #        inverse_func=np.exp
        #    )
        #),
        (
            "TransformedTarget-RandomForestRegressor",
            TransformedTargetRegressor(regressor=RandomForestRegressor(
                n_estimators=100,
                criterion="squared_error",
                bootstrap=True,
                max_samples=0.7,
                oob_score=True,
                n_jobs=-1,
                random_state=874631,
            ),
                transformer=PowerTransformer(
                    method="yeo-johnson",
                    standardize=True,
                    copy=False
                ),
            )
        ),
        #(
        #    "GridSearchCV",
        #    GridSearchCV(
        #        TransformedTargetRegressor(
        #            regressor=MLPRegressor(),
        #            transformer=PowerTransformer(),
        #        ),
        #        param_grid=param_grid,
        #        refit="r2_score",
        #        scoring=scoring,
        #        #n_jobs=-1,
        #        pre_dispatch=4, # avoid jobs explosion
        #        cv=None,        # use default 5 fold cross val,
        #        verbose=4,
        #        return_train_score=True
        #    ),
        #),
        #(
        #    "TransformedTarget-MLPRegressor",
        #    TransformedTargetRegressor(
        #        regressor=MLPRegressor(hidden_layer_sizes=(100,),
        #                               activation="relu",
        #                               solver="sgd",
        #                               alpha=0.0001,
        #                               batch_size="auto",
        #                               learning_rate="constant",
        #                               learning_rate_init=0.001,
        #                               max_iter=500,
        #                               shuffle=True,
        #                               random_state=874631,
        #                               tol=1e-4,
        #                               verbose=True,
        #                               warm_start=False,
        #                               momentum=0.9,
        #                               early_stopping=True,
        #                               validation_fraction=0.1,
        #                               n_iter_no_change=20
        #                               ),
        #        transformer=PowerTransformer(
        #            method="yeo-johnson",
        #            standardize=True,
        #            copy=False
        #        ),
        #    )
        #),
        #(
        #    "MLPRegressor",
        #    MLPRegressor(hidden_layer_sizes=(100,),
        #                 activation="relu",
        #                 solver="sgd",
        #                 alpha=0.0001,
        #                 batch_size="auto",
        #                 learning_rate="constant",
        #                 learning_rate_init=0.001,
        #                 max_iter=200,
        #                 shuffle=True,
        #                 random_state=874631,
        #                 tol=1e-4,
        #                 verbose=True,
        #                 warm_start=False,
        #                 momentum=0.9,
        #                 early_stopping=True,
        #                 validation_fraction=0.1,
        #                 n_iter_no_change=20
        #    )
        #)
        #(
        #    "RandomForestRegressor",
        #    RandomForestRegressor(
        #        n_estimators=100,
        #        criterion="squared_error",
        #        bootstrap=True,
        #        max_samples=0.7,
        #        oob_score=True,
        #        n_jobs=-1,
        #        random_state=874631,
        #    ),
        #),
        #(
        #   "KNeighborsRegressor",
        #   KNeighborsRegressor(
        #       n_neighbors=5,
        #       weights="uniform",
        #       algorithm="auto",
        #       n_jobs=-1,
        #   )
        #),
    ],
    verbose=True,
)

fitting_model = wizard_pipe.fit(X_train, y_train)
pred = wizard_pipe.predict(X_train)
print(
    f"\nExplained variance score is {explained_variance_score(y_true=y_train, y_pred=pred)}",
    f"\nMean Absolute Error is {mean_absolute_error(y_true=y_train, y_pred=pred)}",
    f"\nMean Squared Error is {mean_squared_error(y_true=y_train, y_pred=pred)}",
    f"\nR^2 Error is {r2_score(y_true=y_train, y_pred=pred)}",
)
