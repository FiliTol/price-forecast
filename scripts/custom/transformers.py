import pandas as pd
from geopy.distance import geodesic
from sklearn.base import BaseEstimator, TransformerMixin
import re

pd.options.display.float_format = '{:.0f}'.format


class GeographicTransformer(BaseEstimator, TransformerMixin):
    # https://datascience.stackexchange.com/questions/117200/creating-new-features-as-linear-combination-of-others-as-part-of-a-scikit-learn
    # https://www.andrewvillazon.com/custom-scikit-learn-transformers/
    def __init__(self, locations: dict, column: str = "host_location"):

        self.column = column
        self.locations = locations

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        if self.column == "host_location":
            X = self.transform_to_coordinates(X, self.locations)
            X[self.column] = X.apply(lambda row: self.geodesic_distancer(row, from_loc="host_location"))
            return X
        else:
            X = self.create_strategic_locations_features(X)
            X = self.apply_location_to_feature(X)
            X = self.apply_distancer_to_strategic_locations(X)
            return X

    def transform_to_coordinates(self, X, locations: dict):
        """
        Given an entry and a dictionary, returns the latitude, longitude for
        the entry that are saved in the dictionary
        :param X: dataframe
        :param locations: dict of locations:[latitude, longitude]
        :return: [latitude, longitude]
        """
        try:
            X[self.column] = X[self.column].apply(lambda x: locations.get(x))
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

    @staticmethod
    def create_strategic_locations_features(X: pd.DataFrame) -> pd.DataFrame:
        X["airport_distance_km"] = None
        X["ferretto_square_distance_km"] = None
        X["roma_square_distance_km"] = None
        X["rialto_bridge_distance_km"] = None
        X["san_marco_square_distance_km"] = None
        return X

    def apply_location_to_feature(self, X: pd.DataFrame) -> pd.DataFrame:
        X["airport_distance_km"] = X["airport_distance_km"].apply(lambda x: self.locations["Aeroporto Marco Polo"])
        X["ferretto_square_distance_km"] = X["ferretto_square_distance_km"].apply(
            lambda x: self.locations["Piazza Erminio Ferretto"])
        X["roma_square_distance_km"] = X["roma_square_distance_km"].apply(lambda x: self.locations["Piazzale Roma"])
        X["rialto_bridge_distance_km"] = X["rialto_bridge_distance_km"].apply(
            lambda x: self.locations["Ponte di Rialto"])
        X["san_marco_square_distance_km"] = X["san_marco_square_distance_km"].apply(
            lambda x: self.locations["Piazza San Marco"])
        return X

    def apply_distancer_to_strategic_locations(self, X: pd.DataFrame) -> pd.DataFrame:
        X['airport_distance_km'] = X.apply(lambda row: self.geodesic_distancer(row=row, from_loc="airport_distance_km"),
                                           axis=1)
        X['ferretto_square_distance_km'] = X.apply(
            lambda row: self.geodesic_distancer(row=row, from_loc="ferretto_square_distance_km"), axis=1)
        X['roma_square_distance_km'] = X.apply(
            lambda row: self.geodesic_distancer(row=row, from_loc="roma_square_distance_km"), axis=1)
        X['rialto_bridge_distance_km'] = X.apply(
            lambda row: self.geodesic_distancer(row=row, from_loc="rialto_bridge_distance_km"), axis=1)
        X['san_marco_square_distance_km'] = X.apply(
            lambda row: self.geodesic_distancer(row=row, from_loc="san_marco_square_distance_km"), axis=1)
        return X


class VectorToDataFrame(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        :return: this transform function returns the sparse matrix from tf-idf
        as a list of vectors where every vector is the list of words scores
        """
        dense_matrix = X.toarray()
        combined_column = [dense_matrix[i].tolist() for i in range(dense_matrix.shape[0])]
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
            return '0'
        if "half" in text.lower():
            return '0.5'
        digits = re.findall(r'\d+\.\d+|\d+', str(text))
        return ''.join(digits) if digits else '0'

    @staticmethod
    def remove_digits(text):
        if pd.isna(text):
            return ''
        return re.sub(r'\d', '', str(text)).strip()

    def create_baths_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df['bathrooms'] = df['bathrooms_text'].apply(self.extract_digits)
        df['bathrooms'] = df['bathrooms'].astype(float)
        return df

    def clean_bathrooms_text(self, df: pd.DataFrame) -> pd.DataFrame:
        df['bathrooms_text'] = df['bathrooms_text'].apply(self.remove_digits)
        return df

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X = self.create_baths_column(X)
        X = self.clean_bathrooms_text(X)
        return X.replace(self.mapping)


class CreateStrategicLocationTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Perform arbitary transformation
        X["random_int"] = randint(0, 10, X.shape[0])
        return X


class CreateVerificationsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Perform arbitary transformation
        X["random_int"] = randint(0, 10, X.shape[0])
        return X


class CreateBathsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Perform arbitary transformation
        X["random_int"] = randint(0, 10, X.shape[0])
        return X























