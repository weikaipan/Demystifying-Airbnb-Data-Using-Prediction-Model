"""
Credits:
    https://spark.apache.org/docs/2.2.0/ml-guide.html
    https://stackoverflow.com/questions/35804755/apply-onehotencoder-for-several-categorical-columns-in-sparkmlib
"""


from pyspark import SparkConf
from pyspark.sql import SparkSession, DataFrameWriter, DataFrame
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression, LinearRegressionSummary, DecisionTreeRegressor, GBTRegressor, GeneralizedLinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator 
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Normalizer
from pyspark.ml.feature import VectorIndexer
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.regression import RidgeRegressionWithSGD
from pyspark.mllib.evaluation import RegressionMetrics
# udf
udf_curtodo = udf(lambda cur: cur.replace("$", "").replace(",", ""), StringType())
udf_log = udf(lambda p: log(10, p))
firstelement=udf(lambda v:float(v[0]),FloatType())

# utility functions
def unionAll(*dfs):
        return reduce(DataFrame.unionAll, dfs)

def encoding(df, cols):
    print("Encoding....")
    indexers = [
                    StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
                    for c in cols
    ]

    encoders = [
                    OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol="{0}_encoded".format(indexer.getOutputCol())) 
                    for indexer in indexers
    ]

    assembler = VectorAssembler(inputCols=[
                  encoder.getOutputCol() for encoder in encoders
    ])

    pipeline = Pipeline(stages=indexers + encoders + [assembler])
    return pipeline.fit(df).transform(df)

def assemble(df, cols):
    assembler = VectorAssembler(
        inputCols=cols,
        outputCol="features")
    return assembler.transform(df)

def randomForestRun(train, test, featureIndexer, zillow_test, test_cols):
    print("Training Data Table")
    train.show()
    print("Training...")
    rf = RandomForestRegressor(featuresCol="indexedFeatures")
    pipe = Pipeline(stages=[featureIndexer, rf])
    model = pipe.fit(train)
    print("Training... Done")
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
    r2 = evaluator.evaluate(predictions)
    print("Random Forest Prediction")
    print("R-squared on test data = %g" % r2)
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
    mae = evaluator.evaluate(predictions)
    print("Mean Absolute Error on test data = %g" % mae)
    print("Features Importances")
    k = model.stages[-1].featureImportances
    print(k)

    print("Predicted Price by Zillow")
    zillow_test.show()
    zillow_rdd = zillow_test.rdd
    input_ = zillow_rdd.map(lambda line: (line[0], line[1], line[2], line[3], line[4], line[5], Vectors.dense(line[0:-1])))
    zillow_test = spark.createDataFrame(input_, test_cols + ["features"])
    pred_zillow = model.transform(zillow_test)
    pred_zillow = pred_zillow.withColumn('prediction price/ night', exp(pred_zillow.prediction))
    pred_zillow = pred_zillow.withColumn('prediction price/ month', 30* exp(pred_zillow.prediction))
    pred_zillow.show()
    return pred_zillow

#------------------------------------------------------------------------------------
#   Main function:
#
#------------------------------------------------------------------------------------
spark = SparkSession.builder.master("yarn").appName("Air").config('spark.ui.port', '5420').config('spark.executor.memory', '8g').config('spark.executor.cores', '1').config('spark.cores.max', '1').config('spark.driver.memory','15g').config('spark.num.executors', '5').getOrCreate()
spark.sparkContext.getConf().getAll()

schema = StructType([
            #StructField("neighbourhood", StringType(), True),
            StructField("accommodates", FloatType(), True),
            StructField("bathrooms", FloatType(), True),
            StructField("bedrooms", FloatType(), True),
            StructField("beds", FloatType(), True),
            StructField("cancellation_policy", StringType(), True),
            StructField("extra_people", StringType(), True),
            StructField("id", LongType(), True),
            StructField("instant_bookable", StringType(), True),
            StructField("latitude", FloatType(), True),
            StructField("longitude", FloatType(), True),
            StructField("nta_neighborhood", StringType(), True),
            StructField("price", FloatType(), True),
            StructField("property_type", StringType(), True),
            StructField("review_scores_accuracy", FloatType(), True),
            StructField("review_scores_rating", FloatType(), True),
            
            StructField("room_type", StringType(), True),
            StructField("zipcode", StringType(), True),
 #           StructField("crime_rate", FloatType(), True)
])


zillow_schema = StructType([
    StructField("nta_neighborhood", StringType(), True),
    StructField("property_type", StringType(), True),
    StructField("room_type", StringType(), True),
    StructField("bedrooms", FloatType(), True), 
    StructField("review_scores_rating", FloatType(), True),
    StructField("monthly_rent", FloatType(), True)
])

# # # # # # # # # # # # # # # # # # # # 
# Read and Parse input data           # 
# # # # # # # # # # # # # # # # # # # #
raw_df = spark.read.format("csv").schema(schema).option("header", "true").load("../airbnbZillow.csv")
zillow_df = spark.read.format("csv").schema(zillow_schema).option("header", "true").load("../zillow_total.csv")
print("Reading Complete")

print("Cleaning Data")
raw_df = raw_df.select("price", "accommodates", "bedrooms", "beds", "bathrooms",
                                     "review_scores_rating", "nta_neighborhood","property_type", 
                                     #"review_scores_rating", "crime_rate", "nta_neighborhood","property_type", 
                                      "room_type") 
print("# of lines of raw data:" + str(raw_df.count()))

raw_df = raw_df.dropna()
print("Zillow: " + str(zillow_df.count()))

tmp_df = raw_df.dropDuplicates()
raw_df.unpersist()

df = tmp_df
tmp_df.unpersist()
df.where(df.price == 0).count()
df = df.filter(df.price > 0)
df.select('price').show()

ddf = df.groupby('nta_neighborhood').count()
filter_df = ddf.filter(ddf['count'] > 100)
ddf.unpersist()
filter_df.createOrReplaceTempView('filter_df')
df = df.where('nta_neighborhood IN (SELECT nta_neighborhood FROM filter_df)')
filter_df.unpersist()
df = df.withColumn('price',log(df.price).cast('float')) 
df = df.withColumn('review_scores_rating',(df.review_scores_rating / 10).cast('float')) 
df.show()
print("Cleaning Data ... Done")

# # # # # # # # # # # # # # # # # # # # 
# binary encoding                     # 
# # # # # # # # # # # # # # # # # # # #
cols = ['nta_neighborhood','room_type', 'property_type']
encoded_df = encoding(df, cols)
encoded_df.printSchema()
print("Encoding.... Done!")

print("Running Random Forest: ")
from pyspark.ml.regression import RandomForestRegressor
rf_index = [s + "_indexed" for s in cols]
rf_encoded = [s + "_indexed_encoded" for s in cols]
rf_cols = ['price','bedrooms', 'review_scores_rating']
rf_cols = rf_cols + rf_index
rf_cols_encoded = rf_cols + rf_encoded
rf_df = encoded_df.select(rf_cols)
print("Indexed ..Done")
rf_df_encoded = encoded_df.select(rf_cols_encoded)
rf_df_encoded  = assemble(rf_df_encoded, ["bedrooms", "review_scores_rating", "nta_neighborhood_indexed_encoded", 
                   "room_type_indexed_encoded", "property_type_indexed_encoded"])
print("Encoded ..Done")

print("Pre-Training Data Table")
rf_df.show()
rf_rdd = rf_df.rdd
input_ = rf_rdd.map(lambda line: (line[0],Vectors.dense(line[1:])))
rf_df = spark.createDataFrame(input_, ["label", "features"])
rf_df.printSchema()


print("Zillow Data")

# For Mapping #
nta_map_cols = ['nta_neighborhood', 'nta_neighborhood_indexed']#, 'room_type_indexed']#, 'property_type_indexed']
nta_df = encoded_df.select(nta_map_cols).distinct()
room_map_cols = ['room_type', 'room_type_indexed']
room_df = encoded_df.select(room_map_cols).distinct()
prop_map_cols = ['property_type', 'property_type_indexed']
prop_df = encoded_df.select(prop_map_cols).distinct()

zillow_df = zillow_df.join(nta_df, on='nta_neighborhood', how='left_outer')
zillow_df = zillow_df.join(room_df, on='room_type', how='left_outer')
zillow_df = zillow_df.join(prop_df, on='property_type', how='left_outer')

test_cols = ['bedrooms', 'review_scores_rating', 'nta_neighborhood_indexed', 
             'room_type_indexed', 'property_type_indexed', 'monthly_rent']
zillow_test = zillow_df.select(test_cols)
zillow_test = zillow_test.dropna()

featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures").fit(rf_df)
print("Splitting Data")
(train_rf, test_rf) = rf_df.randomSplit([.7, .3])

print("Prediction (Indexed):")
zillow_pred = randomForestRun(train_rf, test_rf, featureIndexer, zillow_test, test_cols)
zillow_pred = zillow_pred.join(nta_df, on='nta_neighborhood_indexed', how='left_outer')
zillow_pred = zillow_pred.join(room_df, on='room_type_indexed', how='left_outer')
zillow_pred = zillow_pred.join(prop_df, on='property_type_indexed', how='left_outer')
result_cols = ['bedrooms', 'review_scores_rating', 'nta_neighborhood', 'room_type', 
               'property_type', 'prediction price/ night', 'prediction price/ month', 'monthly_rent']
zillow_result = zillow_pred.select(result_cols).sort('monthly_rent', ascending=False)
zillow_result.show()
zillow_result.write.csv('./zillow_prediction_result.csv')
