from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pandas as pd
import json


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
        Given a dict with host locations, saves it to a custom path.
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
