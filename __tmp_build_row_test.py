from dump_market_to_parquet import build_row
rec = {
  'ts': 1234567890,
  'symbol': 'BTCUSDT',
  'bid1': '100.0',
  'ask1': '101.0',
  'bids': '[["100.0","1.23"],["99.9","0.5"]]',
  'asks': '[["101.0","2.0"]]'
}
row = build_row(rec, src='bybit_v5')
print(row)
