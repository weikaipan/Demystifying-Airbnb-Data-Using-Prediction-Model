"""."""
from pyspark.sql import SparkSession
from pyspark.sql.types import *

schema = StructType([StructField("price", FloatType(), True),
                     StructField("accommodates", StringType(), True),
                     StructField("bedrooms", StringType(), True),
                     StructField("beds", StringType(), True),
                     StructField("bathrooms", StringType(), True),
                     StructField("review_scores_rating", FloatType(), True),
                     StructField("property_type", StringType(), True),
                     StructField("room_type", StringType(), True)
                     ])
spark = SparkSession.builder.master("local").appName("Airbnb").getOrCreate()

features = ["accommodates",
            "bedrooms", "beds",
            "bathrooms", "review_scores_rating",
            "property_type", "room_type"]

label = "price"
