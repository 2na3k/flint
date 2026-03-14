"""Distributed Flint demo — run INSIDE the Ray head container.

Setup
-----
1. Start the cluster:
       docker compose up --scale ray-worker=3 -d

2. Wait for all nodes to be healthy:
       docker exec flint-ray-head-1 ray status

3. Run this script from inside the head node:
       docker exec flint-ray-head-1 python /workspace/flint/examples/distributed.py

All paths use /data/ which is volume-mounted on every container and the host.
"""

from __future__ import annotations

import os
import time

import pyarrow as pa
import pyarrow.parquet as pq

# Inside the container /data is the shared volume
DATA_DIR = "/data"
SALES_DIR = os.path.join(DATA_DIR, "sales")
OUTPUT_DIR = os.path.join(DATA_DIR, "sales_summary")

# ---------------------------------------------------------------------------
# 1. Create Hive-partitioned sales dataset on the shared volume
# ---------------------------------------------------------------------------

print("Writing Hive-partitioned sales dataset to /data/sales/ ...")

sales_data = {
    ("year=2023", "region=us"): (
        list(range(1, 1001)),
        [float(i % 200 + 10) for i in range(1000)],
    ),
    ("year=2023", "region=eu"): (
        list(range(1001, 2001)),
        [float(i % 150 + 20) for i in range(1000)],
    ),
    ("year=2024", "region=us"): (
        list(range(2001, 3001)),
        [float(i % 300 + 50) for i in range(1000)],
    ),
    ("year=2024", "region=eu"): (
        list(range(3001, 4001)),
        [float(i % 250 + 30) for i in range(1000)],
    ),
    ("year=2024", "region=apac"): (
        list(range(4001, 5001)),
        [float(i % 180 + 15) for i in range(1000)],
    ),
}

products = ["A", "B", "C"]
total_written = 0
for (year_part, region_part), (order_ids, amounts) in sales_data.items():
    dir_path = os.path.join(SALES_DIR, year_part, region_part)
    os.makedirs(dir_path, exist_ok=True)
    n = len(order_ids)
    table = pa.table(
        {
            "order_id": order_ids,
            "amount": amounts,
            "product": [products[i % 3] for i in range(n)],
        }
    )
    pq.write_table(table, os.path.join(dir_path, "part-0.parquet"))
    total_written += n

print(f"  {total_written} rows across {len(sales_data)} partitions.\n")

# ---------------------------------------------------------------------------
# 2. Connect to the local Ray cluster (inside the container)
# ---------------------------------------------------------------------------

from flint.session import Session  # noqa: E402

print("Connecting to Ray cluster (auto) ...")
session = Session(
    local=False,
    ray_address="auto",  # inside the container: connects to the local cluster
    n_workers=3,
)
print(f"  {session}\n")

t0 = time.time()

# ---------------------------------------------------------------------------
# 3. Read — auto-detects year= / region= Hive partitioning
# ---------------------------------------------------------------------------

df = session.read_parquet(SALES_DIR)

print("== Logical plan ==")
df.explain("logical")

print("\n== Optimized plan after filter(year=2024) ==")
df.filter("year = 2024").explain("optimized")

# ---------------------------------------------------------------------------
# 4. Filter + aggregate — partition pruning skips 2023 files
# ---------------------------------------------------------------------------

print("\n--- Total sales per product in 2024 (3 partitions, pruned from 5) ---")
result = (
    df.filter("year = 2024")
    .groupby("product")
    .agg({"amount": "sum", "order_id": "count"})
    .compute()
)
print(result.to_pandas().sort_values("product").to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Average order value by region in 2024 (excluding apac)
# ---------------------------------------------------------------------------

print("\n--- Average order value by region in 2024 (US vs EU) ---")
result2 = (
    df.filter("year = 2024")
    .filter("region != 'apac'")
    .groupby("region")
    .agg({"amount": "mean"})
    .compute()
)
print(result2.to_pandas().sort_values("region").to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Repartition across 3 workers + map transform
# ---------------------------------------------------------------------------

print("\n--- Distributed map: tag premium orders (amount > 100) ---")


def tag_order(row: dict) -> dict:
    return {**row, "tier": "premium" if row["amount"] > 100 else "standard"}


premium_count = (
    df.filter("year = 2024 AND region = 'us'")
    .repartition(3, partition_by="product")  # hash-shuffle across 3 workers
    .map(tag_order)
    .filter("tier = 'premium'")
    .count()
)
print(f"  Premium orders in 2024 US: {premium_count}")

# ---------------------------------------------------------------------------
# 7. Hive-partitioned write — shuffle by region first, then parallel write
# ---------------------------------------------------------------------------

print(f"\n--- Writing 2024 data as Hive-partitioned Parquet to {OUTPUT_DIR} ---")
(
    df.filter("year = 2024")
    .write_parquet(OUTPUT_DIR, partition_cols=["region"], n_partitions=3)
    .compute()
)

written_dirs = sorted(os.listdir(OUTPUT_DIR))
print(f"  Partitions created: {written_dirs}")

# ---------------------------------------------------------------------------
# 8. Round-trip read to verify
# ---------------------------------------------------------------------------

total_back = session.read_parquet(OUTPUT_DIR).count()
print(f"  Round-trip row count: {total_back}  (expected 3000)")

elapsed = time.time() - t0
print(f"\nTotal time: {elapsed:.2f}s")

session.stop()
print("Done.")
