# demo/demo_pipeline.py
import asyncio
import time
from pipeline.rag_pipeline import RAGPipeline
from config.pipeline_config import PipelineConfig

async def demonstrate_pipeline():
    """Демонстрация работы полного RAG-конвейера"""
    print("🚀 Запуск демонстрации RAG-конвейера")
    print("=" * 50)

    pipeline = RAGPipeline()
    
    # Тестовые вопросы разной сложности
    test_cases = [
        "Что такое машинное обучение?",
        "Объясни разницу между AI и ML",
        "Какие типы нейронных сетей используются в компьютерном зрении?",
        "Что такое RAG архитектура и как она работает?",
        "Расскажи о квантовых вычислениях" # Тема, которой нет в базе знаний
    ]
    for i, question in enumerate(test_cases, 1):
        print(f"\n📝 Тест {i}: {question}")
        print("-" * 40)
        
        start_time = time.time()
        result = await pipeline.process_question(question)
        end_time = time.time()
        
        print(f"✅ Успех: {result['success']}")
        print(f"⏱ Время обработки: {result['processing_time']:.2f}с")
        print(f"🔍 Найдено документов: {len(result['documents'])}")
        print(f"🤖 Ответ: {result['answer']}")
        print(f"📊 Из кэша: {result.get('cached', False)}")
        
        # Показ топ-документа если есть
        if result['documents']:
            best_doc = result['documents'][0]
            print(f"📄 Лучший документ: {best_doc['metadata']['title']}")
            print(f"🎯 Схожесть: {best_doc['similarity_score']:.3f}")
        
        print("-" * 40)

    # Показ метрик системы
    metrics = pipeline.get_metrics()
    print(f"\n📈 Метрики системы:")
    print(f"Всего запросов: {metrics['total_requests']}")
    print(f"Успешных: {metrics['successful_requests']}")
    print(f"Среднее время: {metrics['average_processing_time']:.2f}с")
    print(f"Попадания в кэш: {metrics['cache_hits']}")

if __name__ == "__main__":
    asyncio.run(demonstrate_pipeline())