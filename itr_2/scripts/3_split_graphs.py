import geopandas as gpd
import pandas as pd

### Read in
nodes = gpd.read_file('../data/shapes/newyork_nodes_2020.gpkg') 
edges = gpd.read_file('../data/shapes/newyork_edges_2020.gpkg')
tracts = gpd.read_file('../data/shapes/nyc_tracts.shp')

### Exclude streets that cross tracts
tracts.to_crs('EPSG:2263', inplace=True)
edges.to_crs('EPSG:2263',  inplace=True)
nodes.to_crs("EPSG:2263",  inplace=True)

### A surprise tool that will help us later
edges['ID'] = [f'G{i}' for i in range(len(edges))]
ids = tracts.sjoin(edges)[['GEOID', 'ID']]

### Only keep things that appear exactly once
rel_ids = ids.copy()[~ids['ID'].duplicated(keep=False)]
edges = edges[edges['ID'].isin(ids['ID'].to_list())]
edges = edges.merge(ids, on='ID').drop('ID', axis=1)

### Associate nodes with a tract
nodes = nodes.sjoin(tracts[['GEOID', 'geometry']])

### Save
nodes.to_file('../data/shapes/newyork_nodes_2020_split.gpkg')
edges.to_file('../data/shapes/newyork_edges_2020_split.gpkg')
