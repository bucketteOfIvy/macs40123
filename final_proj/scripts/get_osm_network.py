import click
import osmnx as ox
import geopandas as gpd

@click.command()
@click.option("--year", prompt="Street Network Year", help="Pull the street network of --place from attic data for which year")
@click.option("--place", prompt="Place to pull from", help="OSM name of place to pull street network for")
@click.option("--write-loc", prompt="Where to write the result")
@click.option("--save-type", default=".shp", prompt="Filetype to export to. Default .shp.")
def get_network(year, place, write_loc, save_type):

    # Query network from 2020 at end of year
    date = '{year}-12-31T12:59:59Z'
    ox.settings.overpass_settings = '"[out:json][timeout:{timeout}][date:"' + date + '"]{maxsize}'
    ox.settings.overpass_endpoint = "https://overpass-api.de/api"

    # Get and save the network
    ntwrk = ox.graph_from_place(place, simplify=False)
    ntwrk = ox.graph_to_gdfs(ntwrk, nodes=False)
    ntwrk.to_file(f'{write_loc}.{save_type}')

if __name__ == "__main__":
    get_network()

