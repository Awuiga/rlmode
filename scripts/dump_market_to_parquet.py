# scripts/dump_market_to_parquet.py
import argparse, json, time, os
import pyarrow as pa, pyarrow.parquet as pq
import redis

def main(stream="md:raw", group="dump", consumer="d1", out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    r = redis.Redis.from_url("redis://localhost:6379/0")
    try:
        r.xgroup_create(stream, group, id="0-0", mkstream=True)
    except redis.ResponseError:
        pass

    batch = []
    schema = pa.schema([
        ("ts", pa.int64()),
        ("symbol", pa.string()),
        ("bid1", pa.float64()),
        ("ask1", pa.float64()),
        ("bids", pa.list_(pa.struct([("p", pa.float64()), ("q", pa.float64())]))),
        ("asks", pa.list_(pa.struct([("p", pa.float64()), ("q", pa.float64())]))),
        ("last_price", pa.float64()),
        ("last_side", pa.string()),
        ("last_qty", pa.float64())
    ])
    writer = None
    fname = os.path.join(out_dir, f"md_{int(time.time())}.parquet")

    while True:
        resp = r.xreadgroup(group, consumer, {stream: ">"}, count=200, block=1000)
        if not resp:
            continue
        for _, msgs in resp:
            for msg_id, fields in msgs:
                obj = json.loads(fields[b"data"].decode())
                row = {
                    "ts": obj.get("ts"),
                    "symbol": obj.get("symbol"),
                    "bid1": obj.get("bid1"),
                    "ask1": obj.get("ask1"),
                    "bids": [{"p":p,"q":q} for p,q in obj.get("bids",[])],
                    "asks": [{"p":p,"q":q} for p,q in obj.get("asks",[])],
                    "last_price": (obj.get("last_trade") or {}).get("price"),
                    "last_side":  (obj.get("last_trade") or {}).get("side"),
                    "last_qty":   (obj.get("last_trade") or {}).get("qty"),
                }
                batch.append(row)
                r.xack(stream, group, msg_id)
        if len(batch) >= 5000:
            table = pa.Table.from_pylist(batch, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(fname, schema)
            writer.write_table(table)
            batch.clear()
            print("flushed 5k rows ->", fname)

if __name__ == "__main__":
    main()
