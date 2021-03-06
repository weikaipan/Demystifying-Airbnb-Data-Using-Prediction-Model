"""Script for reading and cleaning data."""

from pyspark.sql.types import *
from pyspark.sql.functions import *
from task_configs import schema

def clean_data(airbnb_df):
    """."""
    airbnb_df = airbnb_df.na.fill({'bathrooms': 'unknown',
                                   'review_scores_rating': 0.0,
                                   'accommodates': 'unknown',
                                   'bedrooms': 'unknown',
                                   'neighbourhood': 'unknown',
                                   'beds': 'unknown',
                                   'property_type': 'unknown',
                                   'room_type': 'unknown',
                                   'price': 0.0,
                                   'last_scraped': 'unknown'})
    # airbnb_df = airbnb_df.filter(airbnb_df.price > 0.0)
    airbnb_df = airbnb_df.orderBy('last_scraped')
    return airbnb_df


def read_data(path, spark):
    """."""
    df = spark.read.load(path, format='csv', header=True, schema=schema)
    df.show()
    print("Data Loaded.")

    return df
