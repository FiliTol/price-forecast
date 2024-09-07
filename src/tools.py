from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pandas as pd
import json
import os
import re


class JsonHandler:
    def __init__(self, user_agent: str = "JsonHandler"):
        """
        Initializes the GeoDataHandler with a user agent for Nominatim.
        :param user_agent: A string representing the user agent for Nominatim.
        """
        self.geolocator = Nominatim(user_agent=user_agent)
        self.geocode = RateLimiter(self.geolocator.geocode, min_delay_seconds=1.1)

    def retrieve_host_location(self, df: pd.DataFrame) -> dict:
        """
        From a dataset of listings, extracts the list of unique host locations
        and retrieve latitude and longitude of every location.
        :param df: pandas DataFrame of listings.
        :return: dict of locations: [latitude, longitude]
        """
        location_geo = {}
        try:
            for location in df["host_location"].unique().tolist():
                host_location = self.geocode(location)
                if host_location:
                    location_geo[location] = (
                        host_location.latitude,
                        host_location.longitude,
                    )
                else:
                    location_geo[location] = (None, None)
            return location_geo
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    @staticmethod
    def export_to_json(dict_object: dict, path: str) -> None:
        """
        Given a dict with host locations, saves it to a src path.
        :param dict_object: dictionary to be saved as JSON.
        :param path: str with the path where to save JSON.
        :return: None
        """
        try:
            with open(path, "w") as f:
                json.dump(dict_object, f)
        except Exception as e:
            print(f"An error occurred while exporting to JSON: {e}")

    @staticmethod
    def import_from_json(path: str) -> dict:
        """
        Import host location from saved JSON.
        :param path: path where the JSON is saved.
        :return: JSON in dictionary form.
        """
        try:
            with open(path, "r") as f:
                dict_object = json.load(f)
            return dict_object
        except Exception as e:
            print(f"An error occurred while importing from JSON: {e}")
            return None


def load_tourism_data():
    """
    Load tourism data from spreadsheet
    :return: tourism dataframe with city acronyms as index
    """
    tourism_city_data = pd.read_excel("data/city_data/urb_ctour_page_spreadsheet.xlsx",
                                      sheet_name="Data",
                                      index_col=0,
                                      )
    col_names_tourism_data = [i.replace(" ", "_") for i in tourism_city_data.columns.tolist()]
    rename_tourism_columns = {}
    loop_index = 0
    for el in tourism_city_data.columns.tolist():
        rename_tourism_columns[el] = col_names_tourism_data[loop_index]
        loop_index += 1
    tourism_city_data.rename(columns=rename_tourism_columns, inplace=True)
    rename_cities = {
        "Venezia": "ve",
        "Milano": "mi",
        "Bergamo": "bg",
        "Roma": "rm",
        "Firenze": "fi",
        "Bologna": "bo",
        "Napoli": "na"
    }
    tourism_city_data.rename(index=rename_cities, inplace=True)

    return tourism_city_data


def add_tourism_data(df: pd.DataFrame):
    """
    Takes a dataframe and adds the touristic features for every city
    :param df: dataframe of listings
    :return: dataframe of listings with touristic features added
    """
    tourism_df = load_tourism_data()
    df["Total_nights_spent_in_tourist_accommodation_establishments"] = tourism_df.loc[tourism_df.index==df["df_city_location"], "Total_nights_spent_in_tourist_accommodation_establishments"]
    df["Nights_spent_in_tourist_accommodation_establishments_by_residents"] = tourism_df.loc[tourism_df.index==df["df_city_location"], "Nights_spent_in_tourist_accommodation_establishments_by_residents"]
    df["Nights_spent_in_tourist_accommodation_establishments_by_non-residents"] = tourism_df.loc[tourism_df.index==df["df_city_location"], "Nights_spent_in_tourist_accommodation_establishments_by_non-residents"]
    df["Total_nights_spent_in_tourist_accommodation_establishments_per_resident_population"] = tourism_df.loc[tourism_df.index==df["df_city_location"], "Total_nights_spent_in_tourist_accommodation_establishments_per_resident_population-residents"]


def concatenate_listings_datasets() -> pd.DataFrame:
    """
    Given a folder path, return the dataframe that is the
    concatenation of the csv in the folder path
    :return: dataframe containing all city dataframes
    """
    datasets = {}

    for file in os.listdir("data/all_cities"):
        pattern = r"_(\w{2})"
        match = re.search(pattern, file)
        result = match.group(1)
        data_frame_prep = pd.read_csv(f"data/all_cities/{file}")
        data_frame_prep["df_city_location"] = file
        data_frame_prep["df_city_location"] = data_frame_prep["df_city_location"].str.slice(start=9, stop=-4)
        datasets[f"df_{result}"] = data_frame_prep
    df = pd.concat([value for key, value in datasets.items()], ignore_index=True)
    return df


def remove_before_double_underscore(input_string):
    result = re.sub(r"^.*?__", "", input_string)
    return result


def return_cleaned_col_names(list_of_names: list) -> list:
    cleaned_names = []
    for name in list_of_names:
        cleaned_names.append(remove_before_double_underscore(name))
    return cleaned_names


def preprocess_text(text):
    text = str(text).lower()
    # punctuation/special char
    text = re.sub(r'[^a-z\s]', '', text)
    # English stop words
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return text

