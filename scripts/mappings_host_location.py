from scripts.custom.tools import JsonHandler
import pandas as pd


handler = JsonHandler()


def load_listings_datasets():
    datasets = {
        "data_dic": pd.read_csv("data/data_dic/d_listings.csv"),
        "data_sep": pd.read_csv("data/data_sep/d_listings.csv"),
        "data_jun": pd.read_csv("data/data_jun/d_listings.csv"),
        "data_mar": pd.read_csv("data/data_mar/d_listings.csv"),
    }
    return datasets


# Could be more efficient by accounting for the locations already retrieved before
def retrieve_host_locations(dataframes):
    host_locations = {
        "data_sep": handler.retrieve_host_location(dataframes["data_sep"]),
        "data_jun": handler.retrieve_host_location(dataframes["data_jun"]),
        "data_mar": handler.retrieve_host_location(dataframes["data_mar"]),
        "data_dic": handler.retrieve_host_location(dataframes["data_dic"]),
    }
    return host_locations


dataframes = load_listings_datasets()
host_locations = retrieve_host_locations(dataframes)
host_locations_total = {
    **host_locations.get("data_sep"),
    **host_locations.get("data_dic"),
    **host_locations.get("data_mar"),
    **host_locations.get("data_jun"),
}
handler.export_to_json(host_locations_total, "data/mappings/host_locations.json")
