import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import urllib.parse
import requests
from io import StringIO

# Read in NYC Crash data using SODA API (my beloathed)
encoded_url = "https://data.cityofnewyork.us/resource/h9gi-nx95.csv?$where=latitude IS NOT NULL AND longitude IS NOT NULL AND crash_date BETWEEN '2020-01-01T00:00:00' AND '2020-12-31T23:59:59'&$limit=50000"
decoded_url = urllib.parse.unquote(encoded_url)
response = requests.get(decoded_url)
data = StringIO(response.text)
crash_data = pd.read_csv(data)

got_bigger, old_size = True, crash_data.shape[0]
while got_bigger:
    encoded_url = f"https://data.cityofnewyork.us/resource/h9gi-nx95.csv?$where=latitude IS NOT NULL AND longitude IS NOT NULL AND crash_date BETWEEN '2020-01-01T00:00:00' AND '2020-12-31T23:59:59'&$limit=50000&$offset={old_size}"
    decoded_url = urllib.parse.unquote(encoded_url)
    response = requests.get(decoded_url)
    data = StringIO(response.text)
    next_crashes = pd.read_csv(data)
    
    crash_data = pd.concat([crash_data, next_crashes], axis=0)

    if crash_data.shape[0] == old_size:
        got_bigger = False

    old_size = crash_data.shape[0]    

# Create a few relevant binaries
crash_data['per_killed'] = crash_data.number_of_persons_killed > 0
crash_data['per_injured'] = crash_data.number_of_persons_injured > 0
crash_data['ped_injured'] = crash_data.number_of_pedestrians_injured > 0 
crash_data['cyc_injured'] = crash_data.number_of_cyclist_injured > 0
crash_data['coords'] = list(zip(crash_data.longitude, crash_data.latitude))
crash_data['coords'] = crash_data.coords.apply(Point)

print("---------------------------------")
print(f"Number of persons injured: {sum(crash_data.per_injured)}")
print(f"Number of pedestrians injured: {sum(crash_data.ped_injured)}")
print(f"Number of cyclists injured: {sum(crash_data.cyc_injured)}")
print("---------------------------------")

# Clean and save
crash_data = crash_data[['per_injured', 'per_killed', 'ped_injured', 'cyc_injured', 'coords']]
crash_data = gpd.GeoDataFrame(crash_data, geometry='coords', crs='EPSG:4326')
crash_data.to_file('../data/shapes/crashes.shp')