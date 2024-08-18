from pandarallel import pandarallel
import pandas as pd
from geopy.distance import geodesic
from sklearn.base import BaseEstimator, TransformerMixin
import re
import reverse_geocode
from typing import Tuple, List

pandarallel.initialize(progress_bar=True)
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


class AmenitiesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, df: pd.DataFrame, remapper: List[Tuple[str, str]]):
        self.amenities_lists: List[str] = df["amenities"].tolist()
        self.remapper: List[Tuple[str, str]] = remapper
        self.amenities_counter: dict = self.get_amenities_counter()
        self.amenities_remapping: dict = self.get_amenities_remapper()

    def get_amenities_counter(self) -> dict:
        """
        Function that unwraps the amenities column elements and
        adds them to a dictionary to count frequencies.
        :return: Dictionary with amenities counts.
        """
        amenities_counter: dict = {}

        for el in self.amenities_lists:
            for e in el.strip("][").split(", "):
                amenity = e.strip('"')
                amenities_counter[amenity] = amenities_counter.get(amenity, 0) + 1
        return amenities_counter

    def get_amenities_remapper(self) -> dict:
        """
        Takes a list of tuples in which every tuple contains (pattern, name)
        and returns a dictionary with the remapped entries.
        :return: Dictionary containing remapped amenities.
        """
        amenities_remapping = {}
        for pattern, name in self.remapper:
            if name == "other":
                regex = re.compile(pattern, re.IGNORECASE)
                for am in self.amenities_counter.keys():
                    if not regex.search(am):
                        amenities_remapping[am] = "other"
            else:
                regex = re.compile(pattern, re.IGNORECASE)
                for am in self.amenities_counter.keys():
                    if regex.search(am):
                        amenities_remapping[am] = name
        return amenities_remapping

    def unwrap_remap_amenities(self, value):
        element = [e.strip('"') for e in value.strip("][").split(", ")]
        remapped_amenities = pd.Series(element).map(self.amenities_remapping)
        return remapped_amenities.tolist()

    @staticmethod
    def return_amenity_counter(row):
        amenities = [
            "AC/heating",
            "technology",
            "kitchen",
            "benefits",
            "toiletry",
            "other",
        ]
        counts = {amenity: row["amenities"].count(amenity) for amenity in amenities}
        for amenity, count in counts.items():
            row[f"amenities_{amenity}"] = count
        return row

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X["amenities"] = X["amenities"].parallel_apply(self.unwrap_remap_amenities)
        X = X.parallel_apply(self.return_amenity_counter, axis=1)
        return X


class OfflineLocationFinder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    @staticmethod
    def retrieve_city(row):
        coords = (row["latitude"], row["longitude"])
        row["listing_city"] = reverse_geocode.get(coords)["city"]
        row["listing_city_pop"] = reverse_geocode.get(coords)["population"]
        return row

    def transform(self, X, y=None):
        X = X.parallel_apply(self.retrieve_city, axis=1)
        return X


class PropertyTypeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, df: pd.DataFrame, remapper: List[Tuple[str, str]]):
        self.property_type_list = df["property_type"].tolist()
        self.remapper: List[Tuple[str, str]] = remapper
        self.property_frequencies = {
            x: self.property_type_list.count(x) for x in self.property_type_list
        }
        self.property_type_remapping = self.property_type_remapper()

    def fit(self, X, y=None):
        return self

    def property_type_remapper(self):
        property_type_remapping = {}
        for pattern, name in self.remapper:
            if name == "other":
                regex = re.compile(pattern, re.IGNORECASE)
                for am in self.property_frequencies.keys():
                    if not regex.search(am):
                        property_type_remapping[am] = "other"
            else:
                regex = re.compile(pattern, re.IGNORECASE)
                for am in self.property_frequencies.keys():
                    if regex.search(am):
                        property_type_remapping[am] = name
        return property_type_remapping

    def transform(self, X, y=None):
        X["property_type"] = X["property_type"].parallel_map(
            self.property_type_remapping
        )
        return X


class HostLocationImputer(TransformerMixin):
    @staticmethod
    def fill_host_location(row):
        if pd.isna(row["host_location"]):
            row["host_location"] = row["listing_city"] + ", Italy"
        return row

    def transform(self, df):
        X = df.copy()
        X = X.parallel_apply(self.fill_host_location, axis=1)
        return X

    def fit(self, *_):
        return self


class ColumnDropperTransformer:
    def __init__(self, columns: list):
        self.columns: list = columns

    def transform(self, X, y=None):
        return X.drop(self.columns, axis=1)

    def fit(self, X, y=None):
        return self


class IntoBinaryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature: str, cat1: str, cond: str, cat2: str):
        self.feature = feature
        self.cat1 = cat1
        self.cond = cond
        self.cat2 = cat2

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self.feature] = X[self.feature].apply(
            lambda x: self.cat1 if eval(self.cond) else self.cat2
        )
        return X
