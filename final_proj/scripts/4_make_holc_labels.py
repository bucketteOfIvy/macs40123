## Author: Ashlynn Wimer
## Date: 12/4/2024
## About: This script creates and attaches HOLC labels to my tract level data.
## The method used was explored in a testing notebook (actually attached in Notebooks/playing.ipynb)

import geopandas as gpd
import pandas as pd
import warnings

### Helper functions for cleaning

def aggregate_yes_no(s: str) -> bool:
    '''
    Return whether a string from the HOLC "negro_yes_or_no" field 
    indicates a "yes" or a "no"
    '''

    # "Nom." is likely a label that Black families are present but
    # "Nominal" in counts. Presence in presence, however, so I'm
    # Counting that as-is

    YES_LST = ['yes', 'few', 'east', 'south', 'west', 'nom.', 'negro', '37', "2"]
    NO_LST  = ['no', '0', 'none']

    if any([y in s.lower() for y in YES_LST]):
        return True

    if any([n in s.lower() for n in NO_LST]):
        return False

    # warnings.warn(f"Unclassified label {s} encountered.")
    return s

def edge_cases(r: pd.Series) -> bool:
    '''
    Fill in hand identified values for missing data in 'negro_yes_or_no' field.
    '''    
    YES_LST = [2672]
    NO_LST = [2739, 2774, 2741, 2745, 2742, 
              2744, 2770, 2778, 2623, 2484, 
              2487, 2496, 2572, 2573, 2588, 
              2743]
    
    if r.get('area_id') in YES_LST:
        return False

    if r.get('area_id') in NO_LST:
        return True

    return r.get('nyn_agg')   


if __name__ == "__main__":
    
    ### Label HOLC zones as either containing Black populations or not.

    ad_data = pd.read_json('../data/shapes/redlining/ad_data.json')

    # Subset to NYC data.
    BOROUGHS = ["Manhattan", "Queens", "Staten Island", "Bronx", "Brooklyn"]
    in_ny = ad_data['state'] == 'NY'
    in_nyc = in_ny & ad_data['city'].isin(BOROUGHS)
    nyc_data = ad_data.copy()[in_nyc]

    # Clean and aggregate the present labels.
    nyc_data['nyn_agg'] = nyc_data.negro_yes_or_no.apply(aggregate_yes_no)
    nyc_data['nyn_agg'] = nyc_data[['area_id', 'nyn_agg']].apply(edge_cases, axis=1)
    if not set(nyc_data.nyn_agg.unique().tolist()) == set((True, False)):
        raise ValueError("Failed to catch some row in neighborhood assignments.")

    # Subset to relevant data.
    nyc_data = nyc_data[['area_id', 'grade', 'nyn_agg']]

    ### Assign tracts to HOLC neighborhoods
    ### We assign each tract to the neighborhood with which it has the highest overlap.

    # GEOID -> (area_id, pct_tract) 
    geoid_to_info = {}

    nyc_crosswalked = gpd.read_file('../data/shapes/redlining/MIv3Areas_2020TractCrosswalk.gpkg')
    nyc_crosswalked = nyc_crosswalked[['area_id', 'GEOID', 'pct_tract']]

    for index, row in nyc_crosswalked.iterrows():
        old_best = geoid_to_info.get(row.GEOID, [0,0])[1]
        if old_best < row.pct_tract:
            geoid_to_info[row.GEOID] = [row.area_id, row.pct_tract]

    # build df
    geoids, area_ids, pct_tracts = [], [], []
    for geoid, (area_id, pct_tract) in geoid_to_info.items():
        geoids.append(geoid)
        area_ids.append(area_id)
        pct_tracts.append(pct_tract * 100)

    ### Bring it all together

    ny_holc_by_largest = pd.DataFrame({'GEOID':geoids, 'area_id':area_ids, "pct_overlap":pct_tracts})
    nyc_holc_by_largest = ny_holc_by_largest.merge(nyc_data, on='area_id', how='inner')

    nyc = gpd.read_file('../data/shapes/nyc_census_imputed.gpkg')
    nyc = nyc.merge(nyc_holc_by_largest[['GEOID', 'grade', 'nyn_agg']], on='GEOID', how='left')
    
    nyc.to_file('../data/shapes/nyc_census_holc.gpkg', index=False)
