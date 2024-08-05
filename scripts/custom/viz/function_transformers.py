import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from scripts.custom.tools import JsonHandler


# Dropping features
def __drop_features_with_many_nan(x: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe, returns the dataframe without the features with at least
    half of the rows Nan
    :param x: dataframe
    :return: dataframe without the partially-Nan columns
    """
    df = x.copy()
    nulls_summary = pd.DataFrame(df.isnull().sum())
    more_than_null_features = nulls_summary.loc[
        nulls_summary.iloc[:, 0] > df.shape[0] * 0.5, :
    ].index.tolist()
    return x.drop(more_than_null_features, axis=1)


fun_tr_drop_features_with_many_nan = FunctionTransformer(__drop_features_with_many_nan)


# Handle long form text features by filling NaNs with empty strings
def __transform_nan_unicode(text_series):
    return text_series.fillna("").astype("U")


fun_tr_transform_nan_unicode = FunctionTransformer(
    __transform_nan_unicode, validate=False
)


# Handle ID data to turn it into strings
def __id_to_string(id_object) -> str:
    return id_object.astype(str)


fun_tr_id_to_string = FunctionTransformer(__id_to_string)


# Handle rates features in string form
def __from_string_to_rate(rate_string) -> float:
    return rate_string.apply(lambda col: col.str.rstrip("%").astype(float))


fun_tr_from_string_to_rate = FunctionTransformer(__from_string_to_rate)


# Handle time features in string form
def __transform_to_datetime(text_date) -> pd.Timestamp | pd.Timestamp:
    return text_date.apply(lambda row: pd.to_datetime(row), axis=1)


fun_tr_transform_to_datetime = FunctionTransformer(__transform_to_datetime)


# Handle price feature in string form
def remove_symbols(text):
    try:
        cleaned_text = re.sub(r"[$,]", "", text)
        return cleaned_text.strip()
    except:
        return None


def remove_dollar_sign(df: pd.DataFrame) -> pd.DataFrame:
    df["price"] = df["price"].apply(remove_symbols).astype(float)
    return df


fun_tr_remove_dollar_sign = FunctionTransformer(remove_dollar_sign)
