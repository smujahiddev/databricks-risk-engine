================================================================================
  PROJECT: allvue-loan-risk-pipeline
================================================================================


────────────────────────────────────────────────────────────────────────────────
  FILE: README.md
────────────────────────────────────────────────────────────────────────────────

# allvue-loan-risk-pipeline

## Overview
Databricks pipeline that ingests raw loan-level data from S3, computes key risk metrics
(LTV ratio, delinquency bucket, expected loss, concentration risk), and writes a risk
summary back to S3 as a partitioned Parquet dataset.

## Folder Structure
```
allvue-loan-risk-pipeline/
├── README.md
├── .gitignore
├── requirements.txt
├── notebooks/
│   ├── 01_ingest_loan_data.py
│   ├── 02_compute_risk_metrics.py
│   └── 03_write_risk_report.py
├── config/
│   └── pipeline_config.json
├── data/
│   └── sample_loans.csv        (local testing only — not committed)
└── utils/
    └── s3_helpers.py
```

## Setup & Run

1. Clone this repo:
   ```bash
   git clone https://github.com/<your-org>/allvue-loan-risk-pipeline.git
   ```

2. Upload config to DBFS:
   ```bash
   databricks fs cp config/pipeline_config.json dbfs:/FileStore/allvue/pipeline_config.json
   ```

3. Place raw loan CSVs in:
   ```
   s3://allvue-data-lake/raw/loans/loans_raw_*.csv
   ```

4. Create a Databricks cluster:
   - Runtime: 13.3 LTS
   - Instance: Standard_DS3_v2 or r5.xlarge (4 cores, 28 GB RAM)
   - Attach IAM role with S3 read/write access

5. Import via Databricks Repos (connect this GitHub repo).

6. Run notebooks in order:
   ```
   01_ingest_loan_data.py
   02_compute_risk_metrics.py
   03_write_risk_report.py
   ```

7. Verify output at:
   ```
   s3://allvue-data-lake/processed/loan_risk/
   ```

## Expected Outputs
- Partitioned Parquet (by region + delinquency bucket) with per-loan risk metrics
- Region concentration summary table
- LTV risk flags (HIGH / MEDIUM / LOW), delinquency buckets, and expected loss per loan


────────────────────────────────────────────────────────────────────────────────
  FILE: requirements.txt
────────────────────────────────────────────────────────────────────────────────

pyspark==3.5.0
boto3>=1.26.0
pytest>=7.0.0
numpy>=1.24.0
pandas>=2.0.0


────────────────────────────────────────────────────────────────────────────────
  FILE: config/pipeline_config.json
────────────────────────────────────────────────────────────────────────────────

{
  "s3_input_bucket": "s3://allvue-data-lake/raw/loans/",
  "s3_output_bucket": "s3://allvue-data-lake/processed/loan_risk/",
  "delinquency_thresholds": {
    "current": 0,
    "30_day": 30,
    "60_day": 60,
    "90_plus_day": 90
  },
  "ltv_high_risk_threshold": 0.80,
  "run_date": "2026-08-24"
}


────────────────────────────────────────────────────────────────────────────────
  FILE: notebooks/01_ingest_loan_data.py
────────────────────────────────────────────────────────────────────────────────

# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Ingest Loan Data from S3
# MAGIC Reads raw CSV loan data from S3 and registers as a Spark temp view.

import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, datediff, current_date

spark = SparkSession.builder.getOrCreate()

# Load config
with open("/dbfs/FileStore/allvue/pipeline_config.json") as f:
    config = json.load(f)

S3_INPUT = config["s3_input_bucket"]

# Read raw loan CSV from S3
df_loans = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(f"{S3_INPUT}loans_raw_*.csv")
)

# Standardize column names
df_loans = (
    df_loans
    .withColumnRenamed("LoanID", "loan_id")
    .withColumnRenamed("BorrowerName", "borrower_name")
    .withColumnRenamed("OriginationDate", "origination_date")
    .withColumnRenamed("MaturityDate", "maturity_date")
    .withColumnRenamed("PrincipalBalance", "principal_balance")
    .withColumnRenamed("PropertyValue", "property_value")
    .withColumnRenamed("LastPaymentDate", "last_payment_date")
    .withColumnRenamed("InterestRate", "interest_rate")
    .withColumnRenamed("LoanType", "loan_type")
    .withColumnRenamed("Region", "region")
)

# Parse dates
df_loans = (
    df_loans
    .withColumn("origination_date", to_date(col("origination_date"), "yyyy-MM-dd"))
    .withColumn("maturity_date", to_date(col("maturity_date"), "yyyy-MM-dd"))
    .withColumn("last_payment_date", to_date(col("last_payment_date"), "yyyy-MM-dd"))
)

df_loans.createOrReplaceTempView("raw_loans")
print(f"Loaded {df_loans.count()} loan records.")
df_loans.printSchema()
display(df_loans.limit(10))


────────────────────────────────────────────────────────────────────────────────
  FILE: notebooks/02_compute_risk_metrics.py
────────────────────────────────────────────────────────────────────────────────

# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Compute Risk Metrics
# MAGIC Calculates LTV, delinquency bucket, expected loss, and concentration risk.

import json
from pyspark.sql.functions import (
    col, when, round as spark_round, datediff,
    current_date, sum as spark_sum, count, lit
)

with open("/dbfs/FileStore/allvue/pipeline_config.json") as f:
    config = json.load(f)

LTV_THRESHOLD = config["ltv_high_risk_threshold"]
thresholds = config["delinquency_thresholds"]

df = spark.table("raw_loans")

# --- 1. Loan-to-Value Ratio ---
df = df.withColumn(
    "ltv_ratio",
    spark_round(col("principal_balance") / col("property_value"), 4)
)

df = df.withColumn(
    "ltv_risk_flag",
    when(col("ltv_ratio") >= LTV_THRESHOLD, "HIGH")
    .when(col("ltv_ratio") >= 0.65, "MEDIUM")
    .otherwise("LOW")
)

# --- 2. Delinquency Bucket ---
df = df.withColumn(
    "days_since_payment",
    datediff(current_date(), col("last_payment_date"))
)

df = df.withColumn(
    "delinquency_bucket",
    when(col("days_since_payment") >= thresholds["90_plus_day"], "90+ DPD")
    .when(col("days_since_payment") >= thresholds["60_day"], "60-89 DPD")
    .when(col("days_since_payment") >= thresholds["30_day"], "30-59 DPD")
    .otherwise("Current")
)

# --- 3. Expected Loss (simplified: PD x LGD x EAD) ---
# Assign PD based on delinquency bucket
df = df.withColumn(
    "pd_estimate",
    when(col("delinquency_bucket") == "90+ DPD", 0.35)
    .when(col("delinquency_bucket") == "60-89 DPD", 0.15)
    .when(col("delinquency_bucket") == "30-59 DPD", 0.05)
    .otherwise(0.01)
)

# LGD assumed 40% (typical for secured loans); EAD = principal_balance
df = df.withColumn("lgd_estimate", lit(0.40))
df = df.withColumn(
    "expected_loss",
    spark_round(col("pd_estimate") * col("lgd_estimate") * col("principal_balance"), 2)
)

# --- 4. Concentration Risk by Region ---
total_balance = df.agg(spark_sum("principal_balance")).collect()[0][0]

region_concentration = (
    df.groupBy("region")
    .agg(
        spark_sum("principal_balance").alias("region_balance"),
        count("loan_id").alias("loan_count")
    )
    .withColumn(
        "concentration_pct",
        spark_round(col("region_balance") / lit(total_balance) * 100, 2)
    )
)

df.createOrReplaceTempView("loan_risk_metrics")
region_concentration.createOrReplaceTempView("region_concentration")

print("Risk metrics computed.")
display(df.select(
    "loan_id", "principal_balance", "ltv_ratio", "ltv_risk_flag",
    "delinquency_bucket", "expected_loss", "region"
).limit(20))

print("\nRegion Concentration:")
display(region_concentration)


────────────────────────────────────────────────────────────────────────────────
  FILE: notebooks/03_write_risk_report.py
────────────────────────────────────────────────────────────────────────────────

# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Write Risk Report to S3

import json
from pyspark.sql.functions import current_timestamp, lit

with open("/dbfs/FileStore/allvue/pipeline_config.json") as f:
    config = json.load(f)

S3_OUTPUT = config["s3_output_bucket"]
RUN_DATE = config["run_date"]

df_risk = spark.table("loan_risk_metrics")
df_region = spark.table("region_concentration")

# Add pipeline metadata
df_risk = df_risk.withColumn("pipeline_run_date", lit(RUN_DATE))
df_risk = df_risk.withColumn("ingested_at", current_timestamp())

# Write loan-level risk report — partitioned by region and delinquency_bucket
(
    df_risk
    .write
    .mode("overwrite")
    .partitionBy("region", "delinquency_bucket")
    .parquet(f"{S3_OUTPUT}loan_level/run_date={RUN_DATE}/")
)

# Write region concentration summary
(
    df_region
    .write
    .mode("overwrite")
    .parquet(f"{S3_OUTPUT}region_concentration/run_date={RUN_DATE}/")
)

print(f"Risk report written to {S3_OUTPUT}")
print(f"Loan-level records: {df_risk.count()}")
print(f"Regions summarized: {df_region.count()}")


────────────────────────────────────────────────────────────────────────────────
  FILE: utils/s3_helpers.py
────────────────────────────────────────────────────────────────────────────────

import boto3
from botocore.exceptions import ClientError


def check_s3_path_exists(bucket: str, prefix: str) -> bool:
    """Check if an S3 path has any objects."""
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return "Contents" in response


def upload_config_to_dbfs(local_path: str, dbfs_path: str):
    """Upload pipeline config to DBFS for notebook access."""
    import subprocess
    subprocess.run(
        ["databricks", "fs", "cp", local_path, dbfs_path, "--overwrite"],
        check=True
    )


def list_s3_partitions(bucket: str, prefix: str):
    """List all partition prefixes under a given S3 path."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    partitions = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            partitions.add(cp["Prefix"])
    return list(partitions)
