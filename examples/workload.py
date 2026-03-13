import os, time, random
os.environ['RAY_LOG_TO_DRIVER'] = '0'

from flint.session import Session

session = Session(local=False, ray_address='auto', n_workers=3)
print('Connected. Starting 30 runs — watch http://localhost:3001', flush=True)

for run in range(1, 31):
    year   = random.choice([2023, 2024, 2025])
    region = random.choice(['us', 'eu', 'apac', 'latam'])
    t0 = time.time()
    result = (
        session.read_parquet('/data/bigdata')
        .filter(f"year = {year} AND region = '{region}'")
        .repartition(3, partition_by='product')
        .map(lambda r: {**r, 'revenue': r['amount'] * (1 + r['score'] / 100)})
        .groupby('product')
        .agg({'revenue': 'sum', 'order_id': 'count'})
        .compute()
    )
    elapsed = time.time() - t0
    groups = result.to_pandas().shape[0]
    print(f'run={run:02d}  year={year}  region={region:<5}  groups={groups}  {elapsed:.2f}s', flush=True)
    time.sleep(0.5)

session.stop()
print('Done.')
