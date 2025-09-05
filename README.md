# HFT Scalping Bot Skeleton

Каркас 24/7 скальпинг-бота для криптовалютных бирж Binance/Bybit. Обучение моделей происходит офлайн, а на VPS выполняется только инференс и исполнение сигналов.

## Структура
- `collector.py` – сбор стакана L2 и трейдов, сохранение в Redis и Parquet.
- `signal_engine.py` – правило‑базированный фильтр сигналов.
- `ai_scorer.py` – инференс seq‑модели (ONNX, CPU).
- `executor.py` – выставление мейкер‑ордров и управление позициями.
- `risk.py` – мониторинг рисков, kill‑switch.
- `monitor.py` – Prometheus метрики и Telegram алерты.
- `config.yml` – параметры торговли и исполнения.
- `risk.yml` – лимиты рисков.
- `docker-compose.yml` – запуск всех компонентов + Redis.

## Локальное обучение моделей
1. Соберите исторические данные с помощью `collector.py`, сохранив в Parquet.
2. На локальном ПК обучите фильтр LightGBM и seq‑модель (1D‑CNN+LSTM).
3. Экспортируйте веса: `model.pkl` (LightGBM), `model.onnx` (seq‑модель).
4. Скопируйте веса и `config.yml`/`risk.yml` на VPS.

## Деплой
```bash
docker-compose build
docker-compose up -d
```
Сервис `redis` хранит очереди и метрики. Конфиги и модели монтируются в контейнеры.

## Автозапуск
Добавьте unit в `systemd` или используйте `restart: unless-stopped` в `docker-compose.yml`.

## Переменные окружения
Файл `.env`:
```
API_KEY=...
API_SECRET=...
TELEGRAM_TOKEN=...
TELEGRAM_CHAT=...
```

## Логи
Все сервисы пишут JSON‑логи в STDOUT. Ротация осуществляется Docker‑движком.

