# tests/test_pipeline.py
import asyncio
import pytest
from pipeline.rag_pipeline import RAGPipeline
from config.pipeline_config import PipelineConfig, RetrieverConfig, GeneratorConfig

class TestRAGPipeline:
    def setup_method(self):
        self.config = PipelineConfig(
            retriever=RetrieverConfig(top_k=3, similarity_threshold=0.3),
            generator=GeneratorConfig(timeout_seconds=10),
            enable_caching=False
        )
        self.pipeline = RAGPipeline(self.config)

    @pytest.mark.asyncio
    async def test_pipeline_success(self):
        """Тест успешного выполнения конвейера"""
        result = await self.pipeline.process_question("Что такое машинное обучение?")
    
        assert result["success"] == True
        assert "answer" in result
        assert "documents" in result
        assert result["processing_time"] > 0
        assert len(result["documents"]) > 0

    @pytest.mark.asyncio
    async def test_pipeline_empty_question(self):
        """Тест обработки пустого вопроса"""
        result = await self.pipeline.process_question("")
    
        assert result["success"] == False
        assert "error" in result["answer"].lower()

    @pytest.mark.asyncio
    async def test_pipeline_unknown_topic(self):
        """Тест вопроса по неизвестной теме"""
        result = await self.pipeline.process_question("Что такое квантовая гравитация?")
        
        # Система должна корректно обработать отсутствие информации
        assert result["success"] == True
        assert len(result["documents"]) == 0 or "не знаю" in result["answer"].lower()

    @pytest.mark.asyncio
    async def test_pipeline_caching(self):
        """Тест работы кэширования"""
        self.pipeline.config.enable_caching = True

        # Первый запрос
        result1 = await self.pipeline.process_question("Что такое ИИ?")
        assert result1["cached"] == False
        
        # Второй идентичный запрос
        result2 = await self.pipeline.process_question("Что такое ИИ?")
        assert result2["cached"] == True
        assert result1["answer"] == result2["answer"]

    def test_metrics_collection(self):
        """Тест сбора метрик"""
        initial_metrics = self.pipeline.get_metrics()
        assert initial_metrics["total_requests"] == 0
        
        # После выполнения запросов метрики должны обновиться
        asyncio.run(self.pipeline.process_question("Тестовый вопрос"))
        
        updated_metrics = self.pipeline.get_metrics()
        assert updated_metrics["total_requests"] > 0
        assert updated_metrics["average_processing_time"] > 0

if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v"])