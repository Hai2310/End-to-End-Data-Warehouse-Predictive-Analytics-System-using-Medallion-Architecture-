from pyspark.sql import functions as F
from pyspark.sql.types import *
from delta.tables import DeltaTable
import random
from datetime import datetime, timedelta

BRONZE_DB = "bronze"
SILVER_DB = "silver"
GOLD_DB = "gold"

START_DATE = datetime(2024, 1, 1)
BATCH_1 = 20000
BATCH_2 = 5000

cities = ["Hanoi", "HCM", "Danang"]
segments = ["regular", "vip", "enterprise"]
payment_types = ["cash", "card", "wallet"]

def generate_trips(start_id, n):
    rows = []
    for i in range(start_id, start_id + n):
        trip_time = START_DATE + timedelta(minutes=random.randint(0, 60 * 24 * 120))
        distance = round(random.uniform(0.5, 35), 2)
        duration = round(distance * random.uniform(1.8, 4.2), 2)
        base_fare = random.choice([8000, 10000, 12000])
        surge = random.choice([1, 1, 1.2, 1.5, 2])
        fare = base_fare + distance * random.randint(7000, 9000) * surge
        rows.append((
            i,
            random.randint(1, 2000),
            random.choice(cities),
            random.choice(segments),
            distance,
            duration,
            base_fare,
            surge,
            round(fare, 2),
            random.choice(payment_types),
            trip_time
        ))
    return rows

schema = StructType([
    StructField("trip_id", LongType()),
    StructField("customer_id", LongType()),
    StructField("city", StringType()),
    StructField("segment", StringType()),
    StructField("distance_km", DoubleType()),
    StructField("duration_min", DoubleType()),
    StructField("base_fare", DoubleType()),
    StructField("surge_multiplier", DoubleType()),
    StructField("fare_amount", DoubleType()),
    StructField("payment_type", StringType()),
    StructField("trip_datetime", TimestampType())
])

df_b1 = spark.createDataFrame(generate_trips(1, BATCH_1), schema)
df_b2 = spark.createDataFrame(generate_trips(BATCH_1 + 1, BATCH_2), schema)

df_b1.write.mode("overwrite").format("delta").saveAsTable(f"{BRONZE_DB}.trips_bronze")
df_b2.write.mode("append").format("delta").saveAsTable(f"{BRONZE_DB}.trips_bronze")

bronze = spark.table(f"{BRONZE_DB}.trips_bronze")

silver = (
    bronze
    .filter(F.col("fare_amount").isNotNull())
    .filter(F.col("fare_amount") > 0)
    .filter(F.col("distance_km") > 0)
    .withColumn("trip_date", F.to_date("trip_datetime"))
    .withColumn("hour", F.hour("trip_datetime"))
    .withColumn("day_of_week", F.dayofweek("trip_datetime"))
    .withColumn("is_weekend", F.col("day_of_week").isin([1, 7]))
    .withColumn("revenue_bucket",
                F.when(F.col("fare_amount") < 100000, "low")
                 .when(F.col("fare_amount") < 300000, "medium")
                 .otherwise("high"))
)

silver.write.mode("overwrite").format("delta").saveAsTable(f"{SILVER_DB}.trips_silver")

dim_date = (
    silver
    .select("trip_date")
    .distinct()
    .withColumn("date_key", F.date_format("trip_date", "yyyyMMdd").cast("int"))
    .withColumn("year", F.year("trip_date"))
    .withColumn("month", F.month("trip_date"))
    .withColumn("day", F.dayofmonth("trip_date"))
)

dim_city = (
    silver
    .select("city")
    .distinct()
    .withColumn("city_key", F.monotonically_increasing_id())
)

dim_customer = (
    silver
    .select("customer_id", "segment")
    .distinct()
    .withColumn("customer_key", F.monotonically_increasing_id())
)

dim_payment = (
    silver
    .select("payment_type")
    .distinct()
    .withColumn("payment_key", F.monotonically_increasing_id())
)

dim_date.write.mode("overwrite").saveAsTable(f"{GOLD_DB}.dim_date")
dim_city.write.mode("overwrite").saveAsTable(f"{GOLD_DB}.dim_city")
dim_customer.write.mode("overwrite").saveAsTable(f"{GOLD_DB}.dim_customer")
dim_payment.write.mode("overwrite").saveAsTable(f"{GOLD_DB}.dim_payment")

fact_source = (
    silver
    .join(dim_customer, ["customer_id", "segment"])
    .join(dim_city, "city")
    .join(dim_date, silver.trip_date == dim_date.trip_date)
    .join(dim_payment, "payment_type")
    .select(
        "trip_id",
        "customer_key",
        "city_key",
        "date_key",
        "payment_key",
        "distance_km",
        "duration_min",
        "fare_amount",
        "base_fare",
        "surge_multiplier",
        "hour",
        "day_of_week",
        "is_weekend",
        "revenue_bucket"
    )
)

fact_table = f"{GOLD_DB}.fact_trips"

if spark.catalog.tableExists(fact_table):
    existing = spark.table(fact_table).select(F.max("trip_id").alias("max_id")).collect()[0]["max_id"]
    fact_incremental = fact_source.filter(F.col("trip_id") > existing)
    delta = DeltaTable.forName(spark, fact_table)
    delta.alias("t").merge(
        fact_incremental.alias("s"),
        "t.trip_id = s.trip_id"
    ).whenNotMatchedInsertAll().execute()
else:
    fact_source.write.format("delta").saveAsTable(fact_table)
