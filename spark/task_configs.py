"""."""
from pyspark.sql import SparkSession
from pyspark.sql.types import *

schema = StructType([StructField("price", FloatType(), True),
                     StructField("accommodates", StringType(), True),
                     StructField("neighbourhood", StringType(), True),
                     StructField("bedrooms", StringType(), True),
                     StructField("beds", StringType(), True),
                     StructField("bathrooms", StringType(), True),
                     StructField("review_scores_rating", FloatType(), True),
                     StructField("property_type", StringType(), True),
                     StructField("room_type", StringType(), True),
                     StructField("last_scraped", DateType(), True)
                     ])
spark = SparkSession.builder.master("local").appName("Airbnb").getOrCreate()

raw_features = ["accommodates", "neighbourhood",
                "bedrooms", "beds",
                "bathrooms", "review_scores_rating",
                "property_type", "room_type", "last_scraped"]

features = ["accommodates", "neighbourhood",
            "bedrooms", "beds",
            "bathrooms", "review_scores_rating",
            "property_type", "room_type"]

label = "price"

# Local Path
local_file = '../data/cleanedlistings.csv'

# Cluster Path
hadoop_root = '/user/wkp219/project/'
hadoop_file = '/user/wkp219/project/cleanedlistings.csv'
