import osmnx as ox
import geopandas as gpd

ntwrk = ox.graph_from_place('City of New York', simplify=False)

ntwrk = ox.graph_to_gdfs(ntwrk, nodes=False)

ntwrk.to_file('../data/shapes/newyork_network_now.shp')