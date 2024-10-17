import geopandas as gpd
import pandas as pd

# Read in data
print('reading data..')
crashes = gpd.read_file('../data/shapes/crashes.shp')
streets = gpd.read_file('../data/shapes/newyork_network_now.shp')[['osmid', 'width', 'geometry']]

print('converting to EPSG 2263')
crashes = crashes.to_crs('EPSG:2263')
streets = streets.to_crs('EPSG:2263')

# Make 50 feet buffer
print('Buffering..')
streets = streets.set_geometry(streets.geometry.buffer(5))

print('Joining..')
streets = streets.sjoin(crashes, predicate='contains')

print('Saving...')
data = pd.DataFrame(streets.drop('geometry', axis=1))

data.to_csv('../data/joined_streets.csv', index=False)

print(sum(data.per_killed))