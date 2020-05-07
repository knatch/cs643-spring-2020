from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.mllib.tree import RandomForestModel
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors

import sys

testFile = "s3://cs643-spring-2020/TestDataset.csv"

conf = (SparkConf().setAppName("Predict wine app"))
sc = SparkContext("local", conf=conf)
sc.setLogLevel("FATAL")
sqlContext = SQLContext(sc)

if len(sys.argv) == 2:
    testFile = sys.argv[1]
print("==== Reading Test Dataset from ====")
print(testFile)

# read trained model from s3 bucket
model = RandomForestModel.load(sc, "s3://cs643-spring-2020/trainedModel.model")
print(model)

# read TestValidation.csv file from s3
df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true', sep=';').load(testFile)  

print("\n\n==== Test dataset ====")
print("Number of rows = %s " % str(df.count()))

outputRdd = df.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[:11])))

predictions = model.predict(outputRdd.map(lambda x: x.features))
labelsAndPredictions = outputRdd.map(lambda lp: lp.label).zip(predictions)

metrics = MulticlassMetrics(labelsAndPredictions)
# Overall statistics
f1Score = metrics.fMeasure()
print("\n\n==== Summary Stats ====")
print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
print("Weighted precision = %s" % metrics.weightedPrecision)
