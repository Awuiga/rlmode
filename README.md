AI Scalping Bot (CPU, 24/7)

Production-oriented modular HFT-ish scalping bot built in Python 3.11+. 
It uses Redis Streams for reliable message delivery, JSON logs to stdout, and a Prometheus exporter. 
Inference runs on CPU via onnxruntime; model training is done offline.

Sections:
- Stack & services
- Quick start
- Configuration
- Offline training & model export
- Dry-run pipeline & tests
- Deployment notes

Stack & services
- Python 3.11+, Pydantic for strict schemas/configs
- Redis Streams for queues (XADD/XREADGROUP/ACK)
- JSON logs to stdout via structlog
- Prometheus metrics exporter in monitor service
- onnxruntime (CPU) for inference
- docker-compose for multi-service deployment

Services
- collector: subscribes to exchange WS, normalizes to MarketEvent, XADD to md:raw
- signal_engine: computes features, applies rules, emits Candidate to sig:candidates
- ai_scorer: batches L2 window to onnxruntime, emits ApprovedSignal to sig:approved
- executor: transforms entry refs to ladder limit orders, post-only, tracks fills, emits Fill to exec:fills
- risk: aggregates daily PnL/series/fill_rate, emits control:events (STOP) when thresholds trip
- monitor: reads metrics:* events and exposes Prometheus HTTP exporter

Quick start
1) Copy env and adjust secrets
   cp .env.example .env

2) Adjust configs in config/ (app.yml, symbols.yml, risk.yml)

3) Start services
   docker compose up -d

4) Check logs
   docker compose logs -f

5) Prometheus metrics (monitor service)
   http://localhost:8000/metrics

Dry run (no exchange)
- Default config uses `exchange: fake` and produces synthetic market data.
- To run a quick local dry-run test without Docker:
  - Install deps (Python 3.11+): `pip install -r requirements.txt`
  - Run e2e pipeline test: `pytest -q`
  - Or run without pytest runner: `python scripts/run_dry.py`
  - Internally uses `fakeredis://` and limits each service via `DRY_RUN_MAX_ITER`.

Run single service locally
- Collector: `REDIS_URL=redis://localhost:6379/0 python -m app.collector.main`
- Signal engine: `python -m app.signal_engine.main`
- AI scorer: `python -m app.ai_scorer.main`
- Executor: `python -m app.executor.main`
- Risk: `python -m app.risk.main`
- Monitor: `python -m app.monitor.main` and visit `/metrics`

Configuration
- config/app.yml: exchange, symbols, redis, thresholds, ladder, stops, AI thresholds, batching
- config/symbols.yml: tick_size, lot_step, min_qty, min_notional per symbol
- config/risk.yml: daily loss limit, max consecutive losses, fill_rate window and min, markout cap
- .env: API keys and Telegram secrets

Redis Streams
- md:raw, sig:candidates, sig:approved, exec:fills, control:events, metrics:*
- Wrapper RedisStream provides xadd (with idempotency key), read_group (auto group create), and ack

Offline training & model export
- LightGBM filter (example): train offline, export to .pkl (models/signal_filter.pkl)
- Sequence model (example): train offline, export to ONNX (models/seq_model.onnx)
- Place artifacts under models/ and restart relevant services

Example: LightGBM filter
- Train any binary classifier on your labeled features.
- Save with pickle:
  import pickle
  pickle.dump(model, open('models/signal_filter.pkl', 'wb'))

Example: Sequence ONNX (PyTorch)
- Train your LSTM/Transformer, then export:
  torch.onnx.export(model, dummy_input, 'models/seq_model.onnx', opset_version=17)

Dry-run pipeline & tests
- Use the fake exchange adapter for collector and executor
- Run tests locally:
  pytest -q
  - The test sets `REDIS_URL=fakeredis://` and `DRY_RUN_MAX_ITER=150`.
  - It also sets `DRY_RUN_FORCE_LOSSES=1` to trigger the risk kill-switch.

Deployment notes
- Use a VPS in a region close to your exchange
- Enable NTP (chrony/systemd-timesyncd)
- Rotate Docker logs via Docker daemon config or external collector
- Keep secrets in .env and do not commit it
