"""A module for preprocessing and training."""
from pyspark.ml.feature import StandardScaler, MinMaxScaler, MaxAbsScaler
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.feature import VectorIndexer, Normalizer

from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.regression import GBTRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

from task_configs import spark


class model():
    """."""
    def __init__(self,
                 label,
                 features,
                 raw_features,
                 standardize=(False, None),
                 nomralized=False,
                 task_type='regression',
                 params=None,
                 category=False,
                 cross_validation=True,
                 ml_algo='',
                 split_ratio=[0.7, 0.3]):
        """Model init."""
        self.label = label
        self.features = features
        self.raw_features = raw_features
        self.standardize = standardize
        self.nomralized = nomralized
        self.task_type = task_type
        self.params = params
        self.category = category
        self.cross_validation = cross_validation
        self.algorithm = ml_algo
        self.split_ratio = split_ratio
        return

    def train_pipeline(self, dataset):
        """Trainer interface."""
        dataset.show()
        # 1. Preprocessing.
        if self.standardize[0] is True:
            scaler = self.get_scaler(self.standardize[1])

        if self.category is True:
            encoding_pipeline = self.category_encoding(dataset)

        dataset = encoding_pipeline.fit(dataset).transform(dataset)
        dataset.printSchema()
        dataset.select('features').show()

        train, test = dataset.randomSplit(self.split_ratio, seed=12345)

        # 2. Algorithm.
        algorithm = self.pick_algorithm(self.algorithm,
                                        features_col='features',
                                        label_col=self.label)

        # 4. Model
        # if self.cross_validation is True:
        #     best_estimator = grid_search()

        model = algorithm.fit(train)
        print("Test")
        test.printSchema()
        print("Train")
        train.printSchema()
        predictions = model.transform(test)
        predictions.show()
        # 4. report and save.
        self.evaluate(predictions)
        return predictions

    def evaluate(self, predictions):
        """."""
        evaluator = RegressionEvaluator(labelCol='price',
                                        predictionCol='prediction',
                                        metricName='r2')
        r2 = evaluator.evaluate(predictions)
        print("r2: %f" % r2)

    def pick_algorithm(self, algorithm, features_col, label_col):
        """."""
        if algorithm == 'linearregression':
            return LinearRegression(featuresCol=features_col,
                                    labelCol=label_col)
        elif algorithm == 'gbtregressor':
            return GBTRegressor(featuresCol=features_col,
                                labelCol=label_col)

    def get_scaler(self, option):
        """Set up scaler for dataset."""
        if option == 'standard':
            scaler = StandardScaler(inputCol="features",
                                    outputCol="scaledFeatures",
                                    withStd=True, withMean=False)
        elif option == 'minmax':
            scaler = MinMaxScaler(inputCol="features",
                                  outputCol="scaledFeatures")
        elif option == 'maxabs':
            scaler = MaxAbsScaler(inputCol="features",
                                  outputCol="scaledFeatures")
        else:
            scaler = None
        return scaler

    def category_encoding(self, dataset):
        """Dummy variable and one hot encoding."""
        def find_categories(types):
            """Only return features, excludes label."""
            categories, continuous = [], []
            for (col, t) in types:
                if col != self.label:
                    if t == 'string':
                        categories.append(col)
                    if t == 'float':
                        continuous.append(col)
            return categories, continuous

        print("Encoding....")
        categories, continuous = find_categories(dataset.dtypes)
        self.categorical_vars = categories
        self.continuous_vars = continuous
        # 1. Get categorical columns.

        indexers = [StringIndexer(inputCol=c,
                    outputCol="{0}_indexed".format(c))
                    for c in categories]

        onehot = [OneHotEncoderEstimator(inputCols=[indexer.getOutputCol()
                  for indexer in indexers],
                  outputCols=["{}_encoded".format(indexer.getOutputCol()) for indexer in indexers])]

        assembler = VectorAssembler(inputCols=[encoder.getOutputCols()
                                    for encoder in onehot][0],
                                    outputCol="categorical_encoded")

        training_assembler = VectorAssembler(inputCols=["categorical_encoded"] + continuous,
                                             outputCol="features")
        pipeline = Pipeline(stages=indexers + onehot + [assembler] + [training_assembler])
        return pipeline

    # def grid_search():
    #     """."""
    #     paramGrid = ParamGridBuilder()\
    #                 .addGrid(lr.regParam, [0.1, 0.01]) \
    #                 .addGrid(lr.fitIntercept, [False, True])\
    #                 .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
    #                 .build()
    #     tvs = TrainValidationSplit(estimator=lr,
    #                                estimatorParamMaps=paramGrid,
    #                                evaluator=RegressionEvaluator(),
    #                                # 80% of the data will be used for training, 20% for validation.
    #                                trainRatio=0.8)
