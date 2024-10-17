from pyspark.sql.types import StringType
from pyspark.sql import SparkSession, DataFrameReader
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import functions as F

# Make session
spark = SparkSession.builder.appName("freq_items_40123").getOrCreate()

# Read in data
crashes = spark.read.option('header', True).csv('../data/joined_streets.csv')
crashes = crashes.drop('width', 'index_right')

tags = spark.read.option('header', True).csv('../data/osm_tags_per_way.csv')
tags = tags.withColumn('osmid', tags.way_id)\
            .drop('way_id')\
            .drop('_c0')\
            .join(crashes, on='osmid')

# Fix some read in errors
tags = tags.withColumn('words',
                        F.split(
                            F.regexp_replace(
                                F.regexp_replace(
                                    F.cast(StringType(), 
                                           F.col('words')), r"[\[\]]", ""),
                                r"\"", ""),
                            ",\s+")
                      )
# Add item to word baskets for each injury. Enforce uniqueness, and fit each seperately so we don't
# find trivial "cyc_inj -> per_inj" associations.
tags_per_injured = tags\
        .withColumn("words",
                    F.when(
                        tags.per_injure != "False",
                        F.concat(F.col('words'), F.array(F.lit("'per_injured'"))))\
                    .otherwise(F.col('words')))\
        .select('osmid', 'words')\
        .withColumn("words", F.array_distinct(F.col("words")))

tags_ped_injured = tags\
        .withColumn("words",
                    F.when(
                        tags.ped_injure != "False",
                        F.concat(F.col('words'), F.array(F.lit("'ped_injured'"))))\
                    .otherwise(F.col('words')))\
        .select('osmid', 'words')\
        .withColumn("words", F.array_distinct(F.col("words")))

tags_cyc_injured = tags\
        .withColumn("words",
                    F.when(
                        tags.cyc_injure != "False",
                        F.concat(F.col('words'), F.array(F.lit("'cyc_injured'"))))\
                    .otherwise(F.col('words')))\
        .select('osmid', 'words')\
        .withColumn("words", F.array_distinct(F.col("words")))

# Fit model
fp = FPGrowth(minSupport=0.01)

# Save FPGrowth for each set
for tags_df, file_name in zip([tags_per_injured, tags_ped_injured, tags_cyc_injured], ['perInj', 'pedInj', 'cycInj']):
    fpm = fp.fit(tags_df.select(tags_df.words.alias('items')))
    
    freq_itemsets = fpm.freqItemsets.sort("freq", ascending=False)
    association_rules = fpm.associationRules.sort("antecedent", "consequent")
    
    freq_itemsets.write.mode('overwrite').save(f'../data/itemsets_and_association_rules/{file_name}freqItemsets.parquet')
    association_rules.write.mode('overwrite').save(f'../data/itemsets_and_association_rules/{file_name}associationRules.parquet')
