# Author: Ashlynn Wimer
# Date: 12/3/2024
# About: This file imputes missing values in our dataset. As those Need Fixed.

from sklearn.impute import KNNImputer
import numpy as np
import geopandas as gpd


def knn_impute_spatially(gdf: gpd.GeoDataFrame, var: str, n_neighbors: int=2, 
                           centroid_crs: str='ESRI:102008') -> gpd.GeoDataFrame:
    '''
    Impute a variable in place using a spatial KNN.

    That is, assign it the average value of the 5 nearest _spatial_ neighbors,
    defined by tract centroids.
    '''
    gdf = gdf.copy()
    old_crs = gdf.crs
    gdf = gdf.to_crs(centroid_crs)

    arr = np.array([gdf[var], gdf.centroid.x, gdf.centroid.y]).T

    imputer = KNNImputer(n_neighbors=n_neighbors)
    new_arr = imputer.fit_transform(arr)

    # Attach our values
    gdf[var] = new_arr[:, 0]
    gdf = gdf.to_crs(old_crs)
    
    return gdf

if __name__ == "__main__":
    # to_impute = ['unemplRate', 'perOwnerOcc', 'medHouseVal', 
    #             'workTravelAv', 'percentNoHs', 'percentBach', 
    #             'perBlack', 'perWithoutIns', 'perOver65', 
    #             'medHouseIncome']

    gdf = gpd.read_file('../data/shapes/nyc_census.gpkg')
    to_impute = gdf.drop(['GEOID', 'geometry', 'numCrashes', 'popDens', 'HRI2010', 'injCrashes', 'deathsCrashes'], axis=1).columns

    for col in to_impute:
        print(f'Imputing {col}')
        gdf = knn_impute_spatially(gdf, col, 5)
    
    gdf.to_file('../data/shapes/nyc_census_imputed.gpkg')