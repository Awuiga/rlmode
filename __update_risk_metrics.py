from pathlib import Path

path = Path('app/risk/main.py')
text = path.read_text()
text = text.replace('Metric(name="trades_total", value=1.0)', 'Metric(name="trades_total", value=1.0)')
text = text.replace('Metric(name="wins_total", value=1.0)', 'Metric(name="trades_won_total", value=1.0, labels={"symbol": fill.symbol})')
text = text.replace('Metric(name="precision", value=precision_val)', 'Metric(name="rolling_precision", value=precision_val)')
text = text.replace('Metric(name="fill_rate_rolling", value=(maker_total / trades_total))', 'Metric(name="fill_rate_rolling", value=(maker_total / trades_total))')
text = text.replace('Metric(name="maker_fraction", value=(maker_total / trades_total))', 'Metric(name="fill_rate_rolling", value=(maker_total / trades_total))')
text = text.replace('Metric(name="avg_markout", value=markout_avg)', 'Metric(name="avg_markout_total", value=markout_avg)')
path.write_text(text)
