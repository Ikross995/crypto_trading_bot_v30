🎯 Что делает система самообучения?

1. Trade Journal (Журнал сделок)

Записывает КАЖДУЮ сделку с полным контекстом:





✅ Цены входа/выхода и P&L



✅ Какой сигнал сработал (EMA, RSI, BB и т.д.)



✅ Уверенность сигнала (confidence)



✅ Рыночный режим (тренд/флэт/волатильный)



✅ Технические индикаторы на момент входа



✅ Почему вышли (TP, SL, разворот сигнала)



✅ Качество сделки (0-1 score)

2. Performance Analyzer (Анализатор)

Находит паттерны успеха:





📊 Какие сигналы работают лучше всего



📊 В каком режиме рынка больше прибыли



📊 Оптимальный порог уверенности



📊 Лучшие причины выхода

3. Adaptive Optimizer (Адаптивный оптимизатор)

Автоматически улучшает параметры:





🎚️ Подстраивает confidence threshold



🎚️ Меняет веса сигналов



🎚️ Оптимизирует пороги режимов (ADX)



🎚️ Адаптирует размер риска

4. Real-Time Adaptation (Адаптация в реальном времени)

Быстро реагирует на изменения:





⚡ Снижает риск при проигрышной серии



⚡ Повышает агрессивность при выигрышах



⚡ Может приостановить торговлю при плохих результатах



🚀 Как это работает?

Процесс обучения:

1. Бот открывает сделку → Trade Journal записывает ВСЕ данные
                          ↓
2. Сделка закрывается → Рассчитывается P&L и Quality Score
                          ↓
3. После N сделок (20+) → Performance Analyzer анализирует паттерны
                          ↓
4. Каждые 24 часа → Adaptive Optimizer улучшает параметры
                          ↓
5. В реальном времени → Real-Time Adaptation делает микро-корректировки


Пример обучения:

День 1:

- Bot confidence threshold: 0.50
- Win Rate: 45%
- Лучший сигнал: EMA Crossover (65% win rate)


День 7 (после 50 сделок):

Optimizer обнаружил:
✅ EMA Crossover работает отлично → увеличил вес сигнала до 1.3x
✅ Win rate низкий → повысил confidence threshold до 0.55
✅ Flat режим прибыльнее → повысил flat_adx_threshold


Результат:

- Bot confidence threshold: 0.55 (было 0.50)
- Win Rate: 58% (было 45%)
- Signal weights adjusted




💻 Использование

Включить самообучение в .env:

# Trade Journal
ENABLE_TRADE_JOURNAL=true
JOURNAL_DIR=data/trade_journal

# Adaptive Optimizer
ENABLE_ADAPTIVE_OPTIMIZER=true
OPTIMIZATION_INTERVAL_HOURS=24
MIN_TRADES_FOR_OPTIMIZATION=20

# Real-Time Adaptation
ENABLE_REALTIME_ADAPTATION=true
PAUSE_ON_LOSS_STREAK=5  # Пауза после 5 убытков подряд


Запуск с самообучением:

# Live торговля с самообучением
python cli.py live --symbols BTCUSDT --timeframe 1m --use-imba --self-learning

# Backtest с самообучением
python cli.py backtest --symbol BTCUSDT --days 30 --use-imba --self-learning




📊 Просмотр результатов

1. Статистика журнала:

from strategy.trade_journal import TradeJournal

journal = TradeJournal()
stats = journal.get_statistics()

print(f"Total Trades: {stats['total_trades']}")
print(f"Win Rate: {stats['win_rate']:.2f}%")
print(f"Average Quality Score: {stats['avg_quality_score']:.2f}")
print(f"Total P&L: ${stats['total_pnl']:.2f}")


Вывод:

Total Trades: 127
Win Rate: 58.27%
Average Quality Score: 0.68
Total P&L: $1,247.50


2. Инсайты обучения:

insights = journal.get_learning_insights()

print("Best Signals:")
for signal in insights['best_signals']:
    print(f"  {signal['signal']}: {signal['win_rate']:.1f}% win rate, ${signal['pnl']:.2f} P&L")

print(f"\nBest Regime: {insights['best_regime']}")
print(f"High Confidence Win Rate: {insights['high_confidence_win_rate']:.1f}%")


Вывод:

Best Signals:
  ema_crossover: 68.5% win rate, $547.30 P&L
  bb_squeeze: 62.1% win rate, $389.20 P&L
  rsi_divergence: 55.3% win rate, $211.00 P&L

Best Regime: trend
High Confidence Win Rate: 71.2%


3. Отчет оптимизатора:

from strategy.adaptive_optimizer import AdaptiveOptimizer

optimizer = AdaptiveOptimizer(journal, config)
print(optimizer.get_optimization_report())


Вывод:

=== Adaptive Optimization Report ===
Total Optimizations: 7
Last Optimization: 2025-10-09 14:30

Recent Changes:
  2025-10-09 14:30 - bt_conf_min: 0.50 -> 0.55
  2025-10-09 14:30 - signal_weights: updated
  2025-10-08 14:30 - trend_adx_threshold: 25.0 -> 23.0
  2025-10-08 14:30 - risk_per_trade_pct: 1.0 -> 1.1


4. Экспорт в CSV для анализа:

journal.export_to_csv("data/all_trades.csv")


Теперь можно открыть в Excel/Google Sheets и анализировать!



🎯 Quality Score (Оценка качества сделки)

Каждая сделка получает оценку 0.0 - 1.0:

Формула:

Base Score: 0.5

+ P&L impact (40%):
  - Прибыльная сделка: +0.0 to +0.4
  - Убыточная сделка: -0.4 to -0.0

+ Exit reason (20%):
  - Take Profit hit: +0.2
  - Signal reversal: +0.1
  - Stop Loss hit: -0.1

+ Signal confidence (20%):
  - High confidence (>0.6): +0.2
  - Low confidence (<0.4): -0.2

+ Signal agreement (10%):
  - 7/9 signals agreed: +0.1
  - 3/9 signals agreed: -0.1

+ Profit capture (10%):
  - Captured 80% of peak: +0.08
  - Captured 20% of peak: +0.02


Примеры оценок:

Отличная сделка (Score: 0.85):

- P&L: +5.2% → +0.21
- Exit: Take Profit → +0.20
- Confidence: 0.72 → +0.09
- Agreement: 8/9 signals → +0.08
- Capture: 75% of peak → +0.07
Total: 0.5 + 0.65 = 0.85 (capped at 1.0)


Плохая сделка (Score: 0.22):

- P&L: -3.8% → -0.15
- Exit: Stop Loss → -0.10
- Confidence: 0.42 → -0.03
- Agreement: 4/9 signals → -0.02
- Capture: N/A (loss) → 0.00
Total: 0.5 - 0.30 = 0.20




⚙️ Параметры оптимизации

1. Confidence Threshold (Порог уверенности)

Логика:





Win Rate > 60% → снижаем порог (больше сделок)



Win Rate < 45% → повышаем порог (более селективно)

Пределы: 0.2 - 0.7

2. Signal Weights (Веса сигналов)

Логика:





Сигнал с win rate 70% → вес 1.4x



Сигнал с win rate 50% → вес 1.0x



Сигнал с win rate 30% → вес 0.6x

3. Regime Thresholds (Пороги режимов)

Логика:





Если тренды прибыльны → снижаем trend_adx_threshold (ловим больше трендов)



Если флэт прибылен → повышаем flat_adx_threshold

4. Risk Parameters (Параметры риска)

Логика:





Quality Score > 0.7 + Win Rate > 55% → риск +10%



Quality Score < 0.4 или Win Rate < 45% → риск -10%

Пределы: 0.3% - 2.0% от депозита



🔥 Real-Time Adaptation (Адаптация в реальном времени)

Мгновенно реагирует на последние 10 сделок:

Плохая серия (Win Rate < 30% за последние 10):

adjustments = {
    "confidence_multiplier": 1.2,  # Требуем на 20% выше confidence
    "risk_multiplier": 0.5,        # Снижаем риск вдвое
    "alert": "Poor recent performance - reducing activity"
}


Отличная серия (Win Rate > 70% за последние 10):

adjustments = {
    "confidence_multiplier": 0.9,  # Разрешаем на 10% ниже confidence
    "risk_multiplier": 1.1,        # Повышаем риск на 10%
    "alert": "Excellent recent performance - slightly more aggressive"
}


Критическая серия (5 убытков подряд):

# ПАУЗА в торговле!
if optimizer.should_pause_trading():
    print("🛑 Trading paused due to loss streak")
    # Бот перестает открывать новые позиции




📁 Структура данных

Trade Journal Files:

data/trade_journal/
├── session_20251009_080000.jsonl  # Сегодняшние сделки
├── session_20251008_080000.jsonl  # Вчерашние сделки
├── session_20251007_080000.jsonl  # Позавчера
└── ...


Optimizer State:

data/optimizer_state.json


{
  "last_optimization": "2025-10-09T14:30:00",
  "parameter_history": [
    {
      "timestamp": "2025-10-09T14:30:00",
      "parameter": "bt_conf_min",
      "old_value": 0.50,
      "new_value": 0.55,
      "reason": "adaptive_optimization"
    }
  ],
  "total_optimizations": 7
}




🎓 Примеры использования

Пример 1: Базовое использование

from strategy.trade_journal import TradeJournal
from strategy.adaptive_optimizer import AdaptiveOptimizer
from core.config import Config

# Инициализация
config = Config()
journal = TradeJournal()
optimizer = AdaptiveOptimizer(journal, config)

# При открытии сделки
trade_record = journal.record_trade_entry(
    trade_id="trade_001",
    symbol="BTCUSDT",
    side="buy",
    entry_price=45000.0,
    quantity=0.01,
    signal_info={
        "type": "ema_crossover",
        "confidence": 0.68,
        "signals_agreed": 7,
        "signals_total": 9
    },
    market_context={
        "regime": "trend",
        "volatility": 0.025,
        "adx": 32.5,
        "rsi": 58.2
    },
    position_params={
        "stop_loss": 44500.0,
        "take_profit": 46000.0,
        "leverage": 5
    }
)

# При закрытии сделки
journal.record_trade_exit(
    trade_id="trade_001",
    exit_price=46200.0,
    exit_reason="take_profit",
    pnl=60.0,  # $60 profit
    max_drawdown=-10.0,
    max_profit=65.0
)

# Автооптимизация
if optimizer.should_optimize():
    optimizations = optimizer.optimize()
    print(f"Applied {len(optimizations)} optimizations")


Пример 2: Мониторинг в реальном времени

from strategy.adaptive_optimizer import RealTimeAdaptation

adaptation = RealTimeAdaptation(journal)

# Перед каждой новой сделкой
adjustments = adaptation.get_real_time_adjustments()

if adjustments:
    print(f"Real-time adjustment: {adjustments['alert']}")
    
    # Применяем корректировки
    required_confidence = config.bt_conf_min * adjustments.get('confidence_multiplier', 1.0)
    position_size = base_size * adjustments.get('risk_multiplier', 1.0)

# Проверка на паузу
if adaptation.should_pause_trading():
    print("⚠️ Trading paused - 5 losses in a row")
    # Останавливаем открытие новых позиций


Пример 3: Анализ после торгового дня

# Получить статистику
stats = journal.get_statistics()
print(f"Today's Performance:")
print(f"  Trades: {stats['total_trades']}")
print(f"  Win Rate: {stats['win_rate']:.2f}%")
print(f"  P&L: ${stats['total_pnl']:.2f}")
print(f"  Quality Score: {stats['avg_quality_score']:.2f}")

# Получить инсайты
insights = journal.get_learning_insights()
print(f"\nWhat worked best:")
for signal in insights['best_signals'][:3]:
    print(f"  {signal['signal']}: {signal['win_rate']:.1f}% win rate")

print(f"\nBest market regime: {insights['best_regime']}")

# Экспортировать для детального анализа
journal.export_to_csv(f"reports/trades_{datetime.now().strftime('%Y%m%d')}.csv")




🏆 Преимущества самообучения

✅ Автоматическая оптимизация

Не нужно вручную подбирать параметры - бот делает это сам!

✅ Адаптация к рынку

Рынок меняется → бот меняется вместе с ним

✅ Защита от убытков

Автоматически снижает риск при плохой серии

✅ Максимизация прибыли

Повышает агрессивность при хорошей серии

✅ Полная прозрачность

Каждая сделка записана, каждое изменение логировано

✅ Накопление опыта

Чем дольше работает бот, тем лучше он торгует!



📈 Ожидаемые результаты

Начальная фаза (0-50 сделок):





Бот собирает данные



Минимальная оптимизация



Win Rate: 45-55%

Фаза обучения (50-200 сделок):





Активная оптимизация параметров



Находит лучшие сигналы



Win Rate: 50-60%

Стабильная фаза (200+ сделок):





Оптимизированные параметры



Адаптация к рынку



Win Rate: 55-65%+



⚠️ Важные замечания

1. Минимум данных

Для оптимизации нужно минимум 20 завершенных сделок. До этого оптимизатор не будет вносить изменения.

2. Интервал оптимизации

По умолчанию оптимизация запускается каждые 24 часа. Можно изменить в конфиге.

3. Безопасные границы

Все параметры имеют безопасные пределы (например, confidence 0.2-0.7) чтобы предотвратить слишком агрессивные изменения.

4. Резервное копирование

Trade Journal автоматически сохраняет все данные в файлы. Можно экспортировать в CSV для анализа.

5. Мониторинг

Регулярно проверяйте optimizer_state.json и логи чтобы понимать какие изменения вносит система.



🚀 Начало работы

Шаг 1: Включите в .env

ENABLE_TRADE_JOURNAL=true
ENABLE_ADAPTIVE_OPTIMIZER=true
ENABLE_REALTIME_ADAPTATION=true


Шаг 2: Запустите бота

python cli.py live --symbols BTCUSDT --use-imba --self-learning


Шаг 3: Подождите 20+ сделок

Бот начнет оптимизацию после накопления данных.

Шаг 4: Мониторьте результаты

# Просмотр статистики
python -c "from strategy.trade_journal import TradeJournal; j = TradeJournal(); print(j.get_statistics())"




📚 Дополнительные ресурсы





Trade Journal API: См. strategy/trade_journal.py



Adaptive Optimizer API: См. strategy/adaptive_optimizer.py



Примеры: См. раздел "Примеры использования" выше



Бот учится. Бот адаптируется. Бот улучшается! 🤖📈💰