import sys
for i, line in enumerate(open("app/collector/main.py", "r", encoding="utf-8"), start=1):
    if line.strip().startswith("async def bybit_stream"):
        print(i)
        break
