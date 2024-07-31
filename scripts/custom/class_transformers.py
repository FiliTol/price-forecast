import pandas as pd
from geopy.distance import geodesic
from sklearn.base import BaseEstimator, TransformerMixin
import re

pd.options.display.float_format = "{:.0f}".format


class GeographicTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, locations: dict, column: str = "host_location"):
        self.column = column
        self.locations = locations

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = self.transform_to_coordinates(X)
        X[self.column] = X.apply(
            lambda row: self.geodesic_distancer(row, from_loc="host_location"), axis=1
        )
        return X

    def transform_to_coordinates(self, X):
        """
        Given an entry and a dictionary, returns the latitude, longitude for
        the entry that are saved in the dictionary
        :param X: dataframe
        :return: dataframe containing updated column
        """
        try:
            X[self.column] = X[self.column].apply(lambda x: self.locations.get(x))
            return X
        except:
            return X

    @staticmethod
    def geodesic_distancer(row, from_loc: str):
        try:
            coords_1 = (row[from_loc][0], row[from_loc][1])
            coords_2 = (row["latitude"], row["longitude"])
            return geodesic(coords_1, coords_2).km
        except:
            return None


class VectorToDataFrame(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        :return: this transform function returns the sparse matrix from tf-idf
        as a list of vectors where every vector is the list of words scores
        """
        dense_matrix = X.toarray()
        combined_column = [
            dense_matrix[i].tolist() for i in range(dense_matrix.shape[0])
        ]
        return pd.Series(combined_column)


class NeighborhoodMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapping):
        self.mapping = mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.replace(self.mapping)


class BathroomsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mapping):
        self.mapping = mapping

    @staticmethod
    def extract_digits(text):
        if pd.isna(text):
            return "0"
        if "half" in text.lower():
            return "0.5"
        digits = re.findall(r"\d+\.\d+|\d+", str(text))
        return "".join(digits) if digits else "0"

    @staticmethod
    def remove_digits(text):
        if pd.isna(text):
            return ""
        return re.sub(r"\d", "", str(text)).strip()

    def create_baths_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df["bathrooms"] = df["bathrooms_text"].apply(self.extract_digits)
        df["bathrooms"] = df["bathrooms"].astype(float)
        return df

    def clean_bathrooms_text(self, df: pd.DataFrame) -> pd.DataFrame:
        df["bathrooms_text"] = df["bathrooms_text"].apply(self.remove_digits)
        return df

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X = self.create_baths_column(X)
        X = self.clean_bathrooms_text(X)
        return X.replace(self.mapping)


class CreateStrategicLocationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, locations: dict):
        self.locations = locations

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = self.allocate_features(X)
        X = self.apply_distancer_to_strategic_locations(X)
        return X

    def allocate_features(self, X):
        X["airport_distance_km"] = pd.Series(
            [self.locations["Aeroporto Marco Polo"]] * X.shape[0]
        )
        X["ferretto_square_distance_km"] = pd.Series(
            [self.locations["Piazza Erminio Ferretto"]] * X.shape[0]
        )
        X["roma_square_distance_km"] = pd.Series(
            [self.locations["Piazzale Roma"]] * X.shape[0]
        )
        X["rialto_bridge_distance_km"] = pd.Series(
            [self.locations["Ponte di Rialto"]] * X.shape[0]
        )
        X["san_marco_square_distance_km"] = pd.Series(
            [self.locations["Piazza San Marco"]] * X.shape[0]
        )
        return X

    def apply_distancer_to_strategic_locations(self, X: pd.DataFrame) -> pd.DataFrame:
        X["airport_distance_km"] = X.apply(
            lambda row: self.geodesic_distancer(
                row=row, from_loc="airport_distance_km"
            ),
            axis=1,
        )
        X["ferretto_square_distance_km"] = X.apply(
            lambda row: self.geodesic_distancer(
                row=row, from_loc="ferretto_square_distance_km"
            ),
            axis=1,
        )
        X["roma_square_distance_km"] = X.apply(
            lambda row: self.geodesic_distancer(
                row=row, from_loc="roma_square_distance_km"
            ),
            axis=1,
        )
        X["rialto_bridge_distance_km"] = X.apply(
            lambda row: self.geodesic_distancer(
                row=row, from_loc="rialto_bridge_distance_km"
            ),
            axis=1,
        )
        X["san_marco_square_distance_km"] = X.apply(
            lambda row: self.geodesic_distancer(
                row=row, from_loc="san_marco_square_distance_km"
            ),
            axis=1,
        )
        return X

    @staticmethod
    def geodesic_distancer(row, from_loc: str):
        coords_1 = (row[from_loc][0], row[from_loc][1])
        coords_2 = (row["latitude"], row["longitude"])
        return geodesic(coords_1, coords_2).km


class CreateVerificationsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = self.new_features_for_verifications(X)
        X = self.apply_on_every_row(X)
        return X.drop(["host_verifications"], axis=1)

    @staticmethod
    def new_features_for_verifications(X: pd.DataFrame) -> pd.DataFrame:
        X["email_verification"] = "f"
        X["phone_verification"] = "f"
        X["work_email_verification"] = "f"
        return X

    @staticmethod
    def allocate_verifications_to_variables(row):
        if "email" in row["host_verifications"]:
            row["email_verification"] = "t"
        if "phone" in row["host_verifications"]:
            row["phone_verification"] = "t"
        if "work_email" in row["host_verifications"]:
            row["work_email_verification"] = "t"
        return row

    def apply_on_every_row(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.apply(self.allocate_verifications_to_variables, axis=1)
