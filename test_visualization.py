#!/usr/bin/env python3
"""
Тест системы визуализации обучения
"""

import sys
import asyncio
from datetime import datetime, timezone
sys.path.append('.')

from strategy.learning_visualizer import LearningVisualizer, LearningSnapshot

async def test_visualization():
    """Тестируем создание dashboard и отчетов."""
    print("🧠 Testing Learning Visualization System...")
    
    # Создаем визуализатор
    visualizer = LearningVisualizer()
    print(f"✅ Visualizer created, output dir: {visualizer.output_dir}")
    
    # Создаем тестовый снимок обучения
    test_snapshot = LearningSnapshot(
        timestamp=datetime.now(timezone.utc),
        iteration=1,
        confidence_threshold=0.45,
        position_size_multiplier=1.0,
        dca_enabled=True,
        total_trades=10,
        win_rate=0.6,
        profit_factor=1.2,
        total_pnl=150.50,
        max_drawdown=-0.05,
        sharpe_ratio=1.1,
        recent_adaptations=[
            {"timestamp": "2024-01-01 12:00:00", "trigger": "test_trigger", "confidence": 0.8}
        ],
        ai_recommendations={"confidence": 0.7, "recommendations": {"test_param": "test_value"}},
        adaptations_count=5,
        last_adaptation_trigger="performance_improvement",
        learning_confidence=0.75
    )
    
    # Добавляем снимок в историю
    visualizer.learning_history.append(test_snapshot)
    
    # Создаем второй снимок для графиков
    test_snapshot2 = LearningSnapshot(
        timestamp=datetime.now(timezone.utc),
        iteration=2,
        confidence_threshold=0.48,
        position_size_multiplier=1.1,
        dca_enabled=True,
        total_trades=12,
        win_rate=0.65,
        profit_factor=1.3,
        total_pnl=200.75,
        max_drawdown=-0.03,
        sharpe_ratio=1.2,
        recent_adaptations=[
            {"timestamp": "2024-01-01 12:05:00", "trigger": "another_trigger", "confidence": 0.9}
        ],
        ai_recommendations={"confidence": 0.8, "recommendations": {"test_param": "test_value2"}},
        adaptations_count=6,
        last_adaptation_trigger="win_rate_optimization",
        learning_confidence=0.80
    )
    
    visualizer.learning_history.append(test_snapshot2)
    
    # Тестируем создание dashboard
    try:
        dashboard_path = await visualizer.create_learning_dashboard()
        print(f"✅ Dashboard created: {dashboard_path}")
        
        # Проверяем что файл создался
        if dashboard_path and visualizer.dashboard_file.exists():
            print(f"✅ Dashboard file exists: {visualizer.dashboard_file}")
            file_size = visualizer.dashboard_file.stat().st_size
            print(f"✅ Dashboard file size: {file_size} bytes")
        else:
            print("❌ Dashboard file not found")
            
    except Exception as e:
        print(f"❌ Dashboard creation failed: {e}")
    
    # Тестируем генерацию отчета
    try:
        report = await visualizer.generate_real_time_report(test_snapshot)
        print("✅ Real-time report generated")
        print(f"Report length: {len(report)} characters")
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
    
    # Тестируем логирование адаптации
    try:
        await visualizer.log_adaptation_event(
            trigger="test_adaptation",
            old_params={"confidence_threshold": 0.45, "position_size": 1.0},
            new_params={"confidence_threshold": 0.48, "position_size": 1.1},
            performance_metrics={"win_rate": 0.6, "profit_factor": 1.2},
            reasoning=["Performance improved", "Market conditions favorable"],
            confidence=0.85
        )
        print("✅ Adaptation event logged")
        
    except Exception as e:
        print(f"❌ Adaptation logging failed: {e}")
    
    # Проверяем созданные файлы
    print(f"\n📁 Files in {visualizer.output_dir}:")
    for file in visualizer.output_dir.iterdir():
        if file.is_file():
            print(f"   📄 {file.name} ({file.stat().st_size} bytes)")
    
    print("\n🎉 Visualization system test completed!")
    return True

if __name__ == "__main__":
    asyncio.run(test_visualization())