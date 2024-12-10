### Author: Ashlynn Wimer
### Date: 12/5/2024
### About: This script creates and street related 311 request counts.

from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import urllib.parse
import requests
from io import StringIO
import math

# We know the total observations from the online web portal!
TOTAL_OBSERVATIONS = 678768
MAX_PER_PULL = 50000

def street_related(s: str) -> bool:
    '''
    Indicate whether a 311 complaint type is street related (True) or
    not (False)
    '''
    if s == 'Homeless Street Condition':
        return False
    
    TERMS = ['parking', 'vehicle', 'traffic', 'street', 'highway', 'bridge', 'tunnel']
    for term in TERMS:
        if term in s.lower():
            return True
        
    return False

def retrieve_one_section(offset: int=0) -> pd.DataFrame:
    '''
    Retrieve 50,000 311s for NYC in 2020, offset by some value and save the result.
    '''
    encoded_url = f"https://data.cityofnewyork.us/resource/erm2-nwe9.csv?$where=latitude IS NOT NULL AND longitude IS NOT NULL AND created_date BETWEEN '2020-01-01T00:00:00' AND '2020-12-31T23:59:59'&$limit=50000&$offset={offset}"
    decoded_url = urllib.parse.unquote(encoded_url)
    response = requests.get(decoded_url)

    data = StringIO(response.text)
    data = pd.read_csv(data)
    
    data.to_csv(f'../data/311_{offset}.csv')

def retrieve_all_311s() -> pd.DataFrame:
    '''
    Retrieve all 311s for 2020 NYC in serial.
    '''
    df = None
    last_size, current_size = -1, 0
    while last_size < current_size:

        encoded_url = f"https://data.cityofnewyork.us/resource/erm2-nwe9.csv?$where=latitude IS NOT NULL AND longitude IS NOT NULL AND created_date BETWEEN '2020-01-01T00:00:00' AND '2020-12-31T23:59:59'&$limit=50000&$offset={current_size}"
        decoded_url = urllib.parse.unquote(encoded_url)
        # created_date BETWEEN %22 2020-01-01T00%3A00%3A00%22%20%3A%3A%
        response = requests.get(decoded_url)

        data = StringIO(response.text)
        data = pd.read_csv(data)

        if df is None:
            df = data.copy()
            current_size = len(df)
            continue
        
        # Update for next loop
        df = pd.concat([df, data], axis=0)
        df.drop_duplicates(inplace=True)    
        last_size = current_size
        current_size = len(df)
        
    return df

def multithreaded_scrape(offsets):
    '''
    Scrape and temporarily save 311s in parallel chunks.
    '''
    with ThreadPoolExecutor(max_workers=7) as executor:
        executor.map(retrieve_one_section, offsets)

if __name__ == "__main__":

    pulls_needed = math.ceil(TOTAL_OBSERVATIONS / MAX_PER_PULL)
    offsets = [i*50000 for i in range(pulls_needed)]

    # multithreaded scrapes
    multithreaded_scrape(offsets)

    files = [f'../data/311_{i * 50000}.csv' for i in range(pulls_needed)]
    dfs = [pd.read_csv(file) for file in files]

    df = pd.concat(dfs, axis=1).drop_duplicates()
    
    # Subset to relevant complaints
    mask = df.complaint_type.apply(street_related)
    df = df[mask]

    # Create gdf for counts
    df['geom'] = list(zip(df.latitude, df.longitude))
    df['geom'] = df.geom.apply(Point)
    gdf = gpd.GeoDataFrame(df, geometry=df.geom, crs='EPSG:4326')
    gdf = gdf[['unique_key', 'geometry']]

    # Find counts
    nyc_tracts = gpd.read_file('../data/shapes/nyc_census_holc.gpkg')
    nyc_geoms = nyc_tracts.copy()[['GEOID', 'geometry']]
    nyc_geoms.to_crs('EPSG:2263', inplace=True)
    gdf.to_crs('EPSG:2263', inplace=True)
    counts = nyc_geoms \
        .sjoin(gdf) \
        .value_counts('GEOID') \
        .reset_index() \
        .rename({'count':'street_311s'}, axis=1)

    # Attach counts
    nyc_tracts = nyc_tracts.merge(counts, on='GEOID', how='left')
    nyc_tracts['street_311s'] = nyc_tracts['street_311s'].fillna(0)

    # Save
    nyc_tracts.to_csv('../data/shapes/nyc_311.gpkg')