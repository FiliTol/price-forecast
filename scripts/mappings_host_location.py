from scripts.custom.tools import JsonHandler
import pandas as pd


handler = JsonHandler()


def load_listings_datasets():
    datasets = {'2023dic': pd.read_csv("data/2023dic/d_listings.csv"),
                '2023sep': pd.read_csv("data/2023sep/d_listings.csv"),
                '2024jun': pd.read_csv("data/2024jun/d_listings.csv"),
                '2024mar': pd.read_csv("data/2024mar/d_listings.csv")}
    return datasets


def retrieve_host_locations(dataframes):
    host_locations = {'2023sep': handler.retrieve_host_location(dataframes['2023sep']),
                      '2024jun': handler.retrieve_host_location(dataframes['2024jun']),
                      '2024mar': handler.retrieve_host_location(dataframes['2024mar']),
                      '2023dic': handler.retrieve_host_location(dataframes['2023dic'])}
    return host_locations


dataframes = load_listings_datasets()
host_locations = retrieve_host_locations(dataframes)
host_locations_total = {
    **host_locations.get('2023sep'),
    **host_locations.get('2023dic'),
    **host_locations.get('2024mar'),
    **host_locations.get('2024jun'),
}
handler.export_to_json(host_locations_total, "data/mappings/host_locations.json")
