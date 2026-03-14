"""Basic Flint example — demonstrates the core DataFrame API."""

import pyarrow as pa

import flint

# ── 1. Start a local session ──────────────────────────────────────────────────
session = flint.Session(local=True, n_workers=2)
print(f"Session: {session}\n")

# ── 2. Create a DataFrame from an in-memory Arrow table ───────────────────────
table = pa.table(
    {
        "id": [1, 2, 3, 4, 5, 6, 7, 8],
        "name": ["alice", "bob", "charlie", "diana", "eve", "frank", "grace", "henry"],
        "dept": ["eng", "eng", "mkt", "mkt", "eng", "hr", "hr", "eng"],
        "score": [92, 85, 78, 90, 88, 70, 75, 95],
        "active": [True, True, True, False, True, True, False, True],
    }
)

df = session.from_arrow(table, n_partitions=2)
print("── Raw data ─────────────────────────────────────────────────────────────")
df.show()

# ── 3. Filter + select (SQL expressions, pushed down to DuckDB) ───────────────
print("\n── Active employees with score ≥ 80 ────────────────────────────────────")
filtered = (
    df.filter("active = true")
    .filter("score >= 80")
    .select("id", "name", "dept", "score")
)
filtered.show()

# ── 4. map — add a computed column ────────────────────────────────────────────
print("\n── With grade column (Python map) ───────────────────────────────────────")


def add_grade(row):
    return {**row, "grade": "A" if row["score"] >= 90 else "B"}


graded = filtered.map(add_grade)
graded.show()

# ── 5. groupby + agg ──────────────────────────────────────────────────────────
print("\n── Avg score per department ─────────────────────────────────────────────")
dept_avg = df.filter("active = true").groupby("dept").agg({"score": "mean"})
dept_avg.show()

# ── 6. SQL escape hatch ───────────────────────────────────────────────────────
print("\n── SQL escape hatch: top 3 scores ───────────────────────────────────────")
top3 = df.sql("SELECT name, score FROM this ORDER BY score DESC LIMIT 3")
top3.show()

# ── 7. Join ────────────────────────────────────────────────────────────────────
dept_table = pa.table(
    {
        "dept": ["eng", "mkt", "hr"],
        "location": ["SF", "NYC", "Austin"],
    }
)
dept_df = session.from_arrow(dept_table, n_partitions=1)

print("\n── Join: employees with department location ─────────────────────────────")
joined = (
    df.filter("active = true")
    .join(dept_df, on="dept", how="inner")
    .select("name", "dept", "score", "location")
)
joined.show()

# ── 8. Repartition + flatmap ──────────────────────────────────────────────────
print("\n── flatmap: each employee → two rows (original + bonus row) ────────────")


def expand(row):
    return [row, {**row, "name": row["name"] + "_bonus", "score": row["score"] + 5}]


expanded = df.filter("score >= 90").flatmap(expand)
expanded.show()

# ── 9. Action methods ─────────────────────────────────────────────────────────
total = df.count()
high_scorers = df.filter("score >= 85").count()
print("\n── Stats ────────────────────────────────────────────────────────────────")
print(f"Total employees : {total}")
print(f"High scorers    : {high_scorers}")

# ── 10. to_pandas ─────────────────────────────────────────────────────────────
pdf = dept_avg.to_pandas().sort_values("dept").reset_index(drop=True)
print("\n── dept_avg as pandas ───────────────────────────────────────────────────")
print(pdf.to_string(index=False))

session.stop()
print("\nDone.")
