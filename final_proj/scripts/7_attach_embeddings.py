### Author: Ashlynn Wimer
### Date: 12/5/2024
### About: This script attaches the embeddings found in the 6_get_embeddings.ipynb 
###        to our cumulative dataset. 

import geopandas as gpd
import pandas as pd

if __name__ == "__main__":
    
    nyc = gpd.read_file('../data/shapes/nyc_311.gpkg', use_arrow=True)
    nyc_embeds = pd.read_csv('../data/models/embedding_data_200.csv')
    nyc_embeds.columns = ["GEOID"] + [f"embed{i}" for i in range(128)]
    nyc_embeds['GEOID'] = nyc_embeds.GEOID.astype(str)

    nyc = nyc.merge(nyc_embeds, on='GEOID', how='left')
    nyc.to_file('../data/shapes/nyc_final.gpkg', engine='pyogrio', use_arrow=True)