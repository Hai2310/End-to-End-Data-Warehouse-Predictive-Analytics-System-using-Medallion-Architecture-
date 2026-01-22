from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from datetime import datetime

GOLD_DB = "gold"

fact = spark.table(f"{GOLD_DB}.fact_trips")

dim_city = spark.table(f"{GOLD_DB}.dim_city")
dim_customer = spark.table(f"{GOLD_DB}.dim_customer")
dim_date = spark.table(f"{GOLD_DB}.dim_date")
dim_payment = spark.table(f"{GOLD_DB}.dim_payment")

df = (
    fact
    .join(dim_city, "city_key")
    .join(dim_customer, "customer_key")
    .join(dim_date, "date_key")
    .join(dim_payment, "payment_key")
)

df_ml = (
    df
    .select(
        "trip_id",
        "fare_amount",
        "distance_km",
        "duration_min",
        "base_fare",
        "surge_multiplier",
        "hour",
        "day_of_week",
        "is_weekend",
        "city",
        "segment",
        "payment_type"
    )
    .withColumnRenamed("fare_amount", "label")
)

index_city = StringIndexer(inputCol="city", outputCol="city_idx", handleInvalid="keep")
index_segment = StringIndexer(inputCol="segment", outputCol="segment_idx", handleInvalid="keep")
index_payment = StringIndexer(inputCol="payment_type", outputCol="payment_idx", handleInvalid="keep")

encoder = OneHotEncoder(
    inputCols=["city_idx", "segment_idx", "payment_idx"],
    outputCols=["city_vec", "segment_vec", "payment_vec"]
)

numeric_features = [
    "distance_km",
    "duration_min",
    "base_fare",
    "surge_multiplier",
    "hour",
    "day_of_week",
    "is_weekend"
]

assembler = VectorAssembler(
    inputCols=numeric_features + ["city_vec", "segment_vec", "payment_vec"],
    outputCol="features"
)

train, test = df_ml.randomSplit([0.8, 0.2], seed=42)

lr = LinearRegression(maxIter=50, regParam=0.1, elasticNetParam=0.0)
rf = RandomForestRegressor(numTrees=80, maxDepth=10, seed=42)

lr_pipeline = Pipeline(stages=[
    index_city,
    index_segment,
    index_payment,
    encoder,
    assembler,
    lr
])

rf_pipeline = Pipeline(stages=[
    index_city,
    index_segment,
    index_payment,
    encoder,
    assembler,
    rf
])

lr_model = lr_pipeline.fit(train)
rf_model = rf_pipeline.fit(train)

pred_lr = lr_model.transform(test)
pred_rf = rf_model.transform(test)

evaluator_rmse = RegressionEvaluator(metricName="rmse")
evaluator_mae = RegressionEvaluator(metricName="mae")
evaluator_r2 = RegressionEvaluator(metricName="r2")

rmse_lr = evaluator_rmse.evaluate(pred_lr)
mae_lr = evaluator_mae.evaluate(pred_lr)
r2_lr = evaluator_r2.evaluate(pred_lr)

rmse_rf = evaluator_rmse.evaluate(pred_rf)
mae_rf = evaluator_mae.evaluate(pred_rf)
r2_rf = evaluator_r2.evaluate(pred_rf)

run_time = datetime.now()

metrics = [
    ("LinearRegression", "RMSE", rmse_lr, run_time),
    ("LinearRegression", "MAE", mae_lr, run_time),
    ("LinearRegression", "R2", r2_lr, run_time),
    ("RandomForest", "RMSE", rmse_rf, run_time),
    ("RandomForest", "MAE", mae_rf, run_time),
    ("RandomForest", "R2", r2_rf, run_time)
]

metrics_schema = StructType([
    StructField("model_name", StringType()),
    StructField("metric_name", StringType()),
    StructField("metric_value", DoubleType()),
    StructField("run_datetime", TimestampType())
])

metrics_df = spark.createDataFrame(metrics, metrics_schema)

metrics_df.write.mode("overwrite").format("delta").saveAsTable(f"{GOLD_DB}.ml_metrics")

predictions = (
    pred_rf
    .select(
        "trip_id",
        F.col("label").alias("y_true"),
        F.col("prediction").alias("y_pred"),
        F.abs(F.col("label") - F.col("prediction")).alias("abs_error")
    )
)

predictions.write.mode("overwrite").format("delta").saveAsTable(f"{GOLD_DB}.fact_prediction")
