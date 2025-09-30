import sys
for i, line in enumerate(open("dump_market_to_parquet.py", "r", encoding="utf-8"), start=1):
    if line.strip().startswith("def build_row("):
        print(i)
        break
