from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col

# get namenode - hdfs getconf -namenodes
NAMENODE = "hdfs://ip-172-31-25-55.ec2.internal"

conf = (SparkConf().setAppName("Train wine app"))
sc = SparkContext("local", conf=conf)
sc.setLogLevel("FATAL")
sqlContext = SQLContext(sc)

# https://spark.apache.org/docs/latest/api/python/pyspark.sql.html?
df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true', sep=';').load(NAMENODE + '/data/TrainingDataset.csv')  
validateDf = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true', sep=';').load(NAMENODE + '/data/ValidationDataset.csv')  

# dropping quality column
newDf = df.select(df.columns[:11])

outputRdd = df.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[:11])))

model = RandomForest.trainClassifier(outputRdd,numClasses=10,categoricalFeaturesInfo={}, numTrees=60, maxBins=32, maxDepth=4, seed=42)
# print(model.toDebugString())

validationOutputRdd = validateDf.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[:11])))

predictions = model.predict(validationOutputRdd.map(lambda x: x.features))
labelsAndPredictions = validationOutputRdd.map(lambda lp: lp.label).zip(predictions)

metrics = MulticlassMetrics(labelsAndPredictions)
# Overall statistics
f1Score = metrics.fMeasure()
print("==== Summary Stats ====")
print("Weighted F(1) Score = %3s" % metrics.weightedFMeasure())
print("Weighted precision = %3s" % metrics.weightedPrecision)

print("\n\n==== Saving model ====")
model.save(sc, 's3://cs643-spring-2020/trainedModel.model')
print("Model Saved successfully")
# print("Test Error = " + str(testErr))
