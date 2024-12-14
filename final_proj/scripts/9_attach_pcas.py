import geopandas as gpd
import pandas as pd
import helpers

finance = helpers.cleanly_read_parquet('../data/finance_pca_results.parquet',
                names=['FinPrec', 'Meh', 'Meh2'])

nyc = gpd.read_file('../data/shapes/nyc_final.gpkg')

nyc['FinPrec'] = finance.FinPrec

nyc.to_file('../data/shapes/nyc_final.gpkg')