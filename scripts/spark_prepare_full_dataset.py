"""Big-data-style data prep for the FULL WildlifeReID-10k dataset (all 22
species, 140,489 images), replacing the pandas+sklearn logic in
build_manifest.py/make_splits.py with PySpark DataFrame operations that run
in parallel across local cores (local[*]).

This is a drop-in replacement for the data-prep stage only. It writes
train.csv / val.csv / test.csv in the exact same (path, species, identity)
schema the existing model/training/inference code already expects, so
src/dataset.py, src/model.py, scripts/train.py, scripts/evaluate.py, and
src/inference_server.py require zero changes.

Usage:
    python scripts/spark_prepare_full_dataset.py
"""
import os

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

METADATA_CSV = "data/raw/wildlifereid10k_metadata.csv"
IMAGES_ROOT = "C:/BigDataFull"
OUT_DIR = "data/processed"
MAX_PER_CLASS = 2500
SEED = 42

if __name__ == "__main__":
    spark = (
        SparkSession.builder.appName("WildlifeReID10k-FullPrep")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )

    df = spark.read.csv(METADATA_CSV, header=True, inferSchema=True)
    df = df.withColumn(
        "abs_path",
        F.concat(F.lit(IMAGES_ROOT + "/"), F.col("path")),
    )
    df = df.withColumn("full_identity", F.concat_ws("_", F.col("species"), F.col("identity")))
    df = df.select(
        F.col("abs_path").alias("path"),
        F.col("species"),
        F.col("full_identity").alias("identity"),
    )

    print("=== Rows per species (full dataset) ===")
    df.groupBy("species").count().orderBy(F.desc("count")).show(30, truncate=False)

    # Cap each species at MAX_PER_CLASS, spreading the cap across identities
    # first (same intent as make_splits.py's cap_species, done as a Spark
    # window operation instead of a pandas loop).
    identity_counts = df.groupBy("species").agg(F.countDistinct("identity").alias("n_identities"))
    df = df.join(identity_counts, on="species")
    df = df.withColumn("per_identity_cap", F.greatest(F.lit(1), (F.lit(MAX_PER_CLASS) / F.col("n_identities")).cast("int")))

    w = Window.partitionBy("species", "identity").orderBy(F.rand(seed=SEED))
    df = df.withColumn("row_in_identity", F.row_number().over(w))
    capped = df.filter(F.col("row_in_identity") <= F.col("per_identity_cap"))

    # If a species still has more than MAX_PER_CLASS after the per-identity
    # cap, trim it down with a random per-species row_number cut.
    w2 = Window.partitionBy("species").orderBy(F.rand(seed=SEED))
    capped = capped.withColumn("row_in_species", F.row_number().over(w2))
    capped = capped.filter(F.col("row_in_species") <= F.lit(MAX_PER_CLASS))
    capped = capped.select("path", "species", "identity")

    print("=== Rows per species (after capping at", MAX_PER_CLASS, ") ===")
    capped.groupBy("species").count().orderBy(F.desc("count")).show(30, truncate=False)

    # Identity-grouped 70/15/15 split: hash each identity to a deterministic
    # bucket so every image for a given animal lands in exactly one split,
    # preventing the same leakage problem the WildlifeReID-10k paper flags
    # for naive random splits.
    bucket = F.pmod(F.hash(F.col("identity"), F.lit(SEED)), F.lit(100))
    capped = capped.withColumn(
        "split",
        F.when(bucket < 70, "train").when(bucket < 85, "val").otherwise("test"),
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    for name in ["train", "val", "test"]:
        part = capped.filter(F.col("split") == name).select("path", "species", "identity")
        pdf = part.toPandas().sample(frac=1, random_state=SEED).reset_index(drop=True)
        out_path = os.path.join(OUT_DIR, f"{name}.csv")
        pdf.to_csv(out_path, index=False)
        print(f"{name}: {len(pdf)} rows -> {out_path}")

    spark.stop()
