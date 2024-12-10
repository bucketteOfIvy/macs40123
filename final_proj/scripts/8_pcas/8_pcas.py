### Author: Ashlynn Wimer
### Date: 12/2/2024
### About: This Python script uses pyspark to create k-means economic environment
###        data. 

import geopandas as gpd
import pandas as pd
from pyspark.sql import SparkSession, Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StandardScaler, PCA
import pyspark.sql.functions as F

def run_pca(features, feature_names, min_components, result_prefix=''):
    # Repeat PCA with one new component every time until 
    # we capture enough variance.
    num_components = min_components
    explains_enough_variance = False
    pca_results = None
    while not explains_enough_variance: 
        # Set up PCA
        pca = PCA(k=num_components, inputCol="features", outputCol='pcaFeatures')
        pca_model = pca.fit(features)

        # Have we explained enough variance?
        expl_vars = pca_model.explainedVariance.toArray()
        total_explained_variance = sum(pca_model.explainedVariance.toArray())
        print(f"Explained {total_explained_variance} with {num_components} components.")
        if total_explained_variance >= 0.8 or float(expl_vars[-1]) < 1/len(feature_names):
            print("Saving principal components.")
            # Save component info
            expl_vars = pca_model.explainedVariance.toArray()
            pca_loadings = pca_model.pc.toArray()
            rows = [(i, expl_vars[i], *pca_loadings[:, i]) for i in range(num_components)]
            loading_df = pd.DataFrame(rows, columns=['component', 'explained_var'] + feature_names)
            loading_df.to_csv(f'../../data/{result_prefix}_pca_loadings.csv', header=True)


            # Save transformed data
            pca_results = pca_model.transform(features).select('pcaFeatures')
            pca_results.write.mode('overwrite').parquet(f'../../data/{result_prefix}_pca_results.parquet')

            # End loop
            explains_enough_variance = True
        
        # Prepare for next loop
        num_components += 1

spark = SparkSession \
            .builder \
            .appName('to_finiteness_and_before') \
            .getOrCreate()

#### Preprocessing

# I have a *logic* connecting each of these features to risk of additional
# negative crash outcomes. Hence this approach!
pca_feature_cols = ["perBlack", "unemplRate", "percentNoHs", 'perWithoutIns', "medHouseIncome"]
financial_pca_cols = ["medHouseIncome", "unemplRate", 'perOwnerOcc', 'percentPov', 'percentCarCom', 'percentRentBurd']
med_fric_pca_cols = ['percentNoHs', 'percentBach', 'perBlack', 'perWithoutIns', 'perOver65']
all_cols = list(set(pca_feature_cols + financial_pca_cols + med_fric_pca_cols))

#### Preprocessing

# Read in our data
# It's stored as a geodatapackage, which is not easily readable in pyspark 
df = pd.DataFrame(gpd.read_file('../../data/shapes/nyc_census_imputed.gpkg').drop('geometry', axis=1))
df = spark.createDataFrame(df)

# Get our feature columns
df_features = df.select(*(F.col(c).cast("float").alias(c) for c in all_cols)) \
                .dropna()\
                .withColumn('pca_features', F.array(*[F.col(c) for c in pca_feature_cols])) \
                .withColumn('fin_pca_features', F.array(*[F.col(c) for c in financial_pca_cols])) \
                .withColumn('medfric_pca_features', F.array(*[F.col(c) for c in med_fric_pca_cols])) \
                .select('pca_features', 'fin_pca_features', 'medfric_pca_features')

standardizer = StandardScaler(inputCol = 'features_unscaled', outputCol='features')

pca_vectors = df_features.rdd.map(lambda row: Vectors.dense(row.pca_features))
pca_features = spark.createDataFrame(pca_vectors.map(Row), ["features_unscaled"])
model = standardizer.fit(pca_features)
pca_features = model.transform(pca_features).select('features')

fin_pca_vectors = df_features.rdd.map(lambda row: Vectors.dense(row.fin_pca_features))
fin_pca_features = spark.createDataFrame(fin_pca_vectors.map(Row), ["features_unscaled"])
model = standardizer.fit(fin_pca_features)
fin_pca_features = model.transform(fin_pca_features).select('features')

medfric_pca_vectors = df_features.rdd.map(lambda row: Vectors.dense(row.medfric_pca_features))
medfric_pca_features = spark.createDataFrame(medfric_pca_vectors.map(Row), ["features_unscaled"])
model = standardizer.fit(medfric_pca_features)
medfric_pca_features = model.transform(medfric_pca_features).select('features')

pca_features.persist()
fin_pca_features.persist()
medfric_pca_features.persist()

run_pca(pca_features, pca_feature_cols, 2, "")
run_pca(fin_pca_features, financial_pca_cols, 2, "finance")
run_pca(medfric_pca_features, med_fric_pca_cols, 2, "medical_friction")

