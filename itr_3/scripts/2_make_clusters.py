### Author: Ashlynn Wimer
### Date: 12/2/2024
### About: This Python script uses pyspark to create k-means economic environment
###        data. 

import geopandas as gpd
import pandas as pd
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.mllib.feature import StandardScaler as StandardScalerRDD
from pyspark.mllib.linalg.distributed import RowMatrix
import pyspark.sql.functions as F

spark = SparkSession \
            .builder \
            .appName('to_finiteness_and_before') \
            .getOrCreate()

#### Preprocessing

# I have a *logic* connecting each of these features to risk of additional
# negative crash outcomes. Hence this approach!
pca_feature_cols = ["perBlack", "unempl", "percentNoHs", 
                'perWithoutIns', "perOver65", "perUnder5", 
                "medHouseVal", "medHouseIncome"]

# I have a slightly different outcome of interest on the cluster data.
cluster_feature_cols = pca_feature_cols + ['numCrashes', 'HCI2010']

# Read in our data
# It's stored as a geodatapackage, which is not easily readable in pyspark 
df = pd.DataFrame(gpd.read_file('../data/shapes/nyc_census.gpkg').drop('geometry', axis=1))
df = spark.createDataFrame(df[cluster_feature_cols])

# Get our feature columns
df_features = df.select(*(F.col(c).cast("float").alias(c) for c in feature_cols), "GEOID") \
                .dropna()\
                .withColumn('pca_features', F.array(*[F.col(c) for c in pca_feature_cols])) \
                .withColumn("cluster_features", F.array(*[F.col(c) for c in cluster_feature_cols])) \
                .select('GEOID', 'pca_features', 'cluster_features')

standardizer = StandardScaler(inputCol = 'features_unscaled', outputCol='features')

pca_vectors = df_features.rdd.map(lambda row: Vectors.dense(row.pca_features))
pca_features = spark.createDataFrame(pca_vectors.map(Row), ["features_unscaled"])
model = standardizer.fit(pca_features)
pca_features = model.transform(pca_features).select('features')

cluster_vectors = df_features.rdd.map(lambda row: Vectors.dense(row.cluster_features))
cluster_features = spark.createDataFrame(cluster_vectors.map(Row), ["features_unscaled"])
model = standardizer.fit(cluster_features)
cluster_features = model.transform(cluster_features).select('features')

pca_features.persist()
cluster_features.persist()

### Principal Component Analysis
print("\n-------------- PCA --------------")

# Our approach is to:
# 1. Take a PCA. We're hoping for a good first or second component.
# 2. Cluster without PCA.
# 3. Cluster on our PCAs.

# Repeat PCA with one new component every time until 
# we capture enough variance.
num_components = 2
last_explained, explains_enough_variance = float('-inf'), False
pca_results = None
while not explains_enough_variance: 
    # Set up PCA
    pca = PCA(k=num_components, inputCol="features", outputCol='pcaFeatures')
    pca_model = pca.fit(pca_features)

    # Have we explained enough variance?
    explained_variance = sum(pca_model.explainedVariance.toArray())
    print(f"Explained {explained_variance} with {num_components} components.")
    if explained_variance >= 0.8:
        print("Saving principal components.")
        # Save component info
        expl_vars = pca_model.explainedVariance.toArray()
        rows = [(i, expl_vars[i], *pca_model.pc[i, :]) for i in range(num_components)]
        loading_df = spark.createDataFrame(rows, ['component', 'explained_var'] + pca_feature_cols)
        loading_df.write.csv('../data/pca_loadings.csv', header=True)

        # Save transformed data
        pca_results = model.transform(pca_features).select("pcaFeatures")
        pca_results.write.csv('../data/pca_features.csv')

        # End loop
        explains_enough_variance = True
    
    # Prepare for next loop
    num_components += 1

# Surprise tool that will help us later
pca_features = pca_results.rdd.map(lambda row: Vectors.dense(row.pcaFeatures))
pca_features = spark.CreateDataFrame(pca_features.map(Row), ['features'])

pca_features.persist()

### Cluster without PCAs
# A priori, I'm unwilling to attempt to name more than 5 clusters.
# So I'm going to cluster up to 5 and save the one with the best Silhouette.
print("\n-------------- Clustering not on PCAs --------------")

# I'm writing as though space is expensive but computation is cheap
# So we repeat a computation instead saving the results of the computation.

def fit_transform(cluster_features, cluster_model, num_clusters, save_centers=False):
    '''
    Given the type of cluster to fit, the class of model to fit, and 
    the number of clusters to fit, a clustering model and return the 
    result.
    '''
    kmeans = cluster_model(k=num_clusters)
    model = kmeans.fit(cluster_features)

    if save_centers:
        print("Saving clusters centers..")
        centers = model.clusterCenters
        centers = spark.createDataFrame(centers, [f'Center {i}' for i in range(num_clusters)])
        centers.write.csv('../data/cluster_centers.csv')

    return model.transform(cluster_features)

def find_best_clusters(cluster_features, cluster_model=KMeans, max_k=5):
    '''
    Given cluster features, cluster models, and the max number of clusters
    to test, find the best cluster amount and Silhouette per the Silhouette alone.
    '''
    best_k, best_sil = None, float('-inf')
    for num_clusters in range(2, max_k + 1):
        predictions = fit_transform(cluster_features, cluster_model, num_clusters)

        evaluator = ClusteringEvaluator()
        silhouette = evaluator.evaluate(predictions)
        print(f"{num_clusters} KMean Clusters have silhouette {silhouette}")

        if silhouette > best_sil:
            print("New Best!")
            best_sil = silhouette
            best_k = num_clusters

    return(best_k, best_sil)

best_k, _ = find_best_clusters(cluster_features, KMeans)
fit_transform(cluster_features, KMeans, best_k, save_centers=True)

### Cluster on PCAs
# Cluster again, but on PCAs this time!
# We only even look at these if the PCAs are all somewhat nameable.
print("\n-------------- Clustering on PCAs --------------")

best_k, _ = find_best_clusters(pca_features, KMeans)
fit_transform(pca_features, KMeans, best_k, save_centers=True)

