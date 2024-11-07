import osmnx as ox
import geopandas as gpd
import pygris
import pandas as pd

# Set date for network 
date = f'2020-12-28T00:00:00Z'
ox.settings.overpass_settings = '[out:json][timeout:{timeout}][date:"' + date + '"]{maxsize}'
ox.settings.overpass_url = "https://overpass-api.de/api/interpreter"

# Get the network -- simplified this time!
ntwrk = ox.graph_from_place('City of New York', simplify=True)

nodes, edges = ox.graph_to_gdfs(ntwrk)

edges.to_file('../data/shapes/newyork_edges_2020.gpkg')
nodes.to_file('../data/shapes/newyork_nodes_2020.gpkg')

counties = ['New York', 'Kings', 'Bronx', 'Richmond', 'Queens']
nyc_tracts = [pygris.tracts(state='New York', county=county) \
              for county in counties]
nyc_tracts = gpd.GeoDataFrame (
    pd.concat(nyc_tracts, ignore_index=True)
)

print(nyc_tracts)

nyc_tracts.to_file('../data/shapes/nyc_tracts.shp')