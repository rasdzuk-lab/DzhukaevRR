# evaluation/rag_evaluator.py
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Any
import logging
import asyncio
import os

from .test_dataset import TestCase, EvaluationDataset
from .retrieval_evaluator import RetrievalEvaluator
from .generation_evaluator import GenerationEvaluator
from pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self, rag_pipeline: RAGPipeline):
        self.pipeline = rag_pipeline
        self.retrieval_evaluator = RetrievalEvaluator(rag_pipeline.retriever)
        self.generation_evaluator = GenerationEvaluator()
        self.dataset = EvaluationDataset()
    
    async def run_comprehensive_evaluation(self, test_cases: List[TestCase] = None) -> Dict[str, Any]:
        """Запуск комплексной оценки RAG-системы"""
        if test_cases is None:
            test_cases = self.dataset.test_cases 
        
        logger.info(f"Starting comprehensive evaluation with {len(test_cases)} test cases")
    
        # Этап 1: Оценка ретривера
        retrieval_results = self.retrieval_evaluator.evaluate_dataset(test_cases)
    
        # Этап 2: Получение ответов от полной системы
        generated_answers = []
        loop = asyncio.get_event_loop()
        for test_case in test_cases:
            try:
                result = await self.pipeline.process_question(test_case.question)
                generated_answers.append(result["answer"])
            except Exception as e:
                logger.error(f"Error processing question: {test_case.question}, error: {e}")
                generated_answers.append("") # Пустой ответ в случае ошибки
                
        # Этап 3: Оценка генерации
        generation_results = self.generation_evaluator.evaluate_answers(
            test_cases, generated_answers
        )
    
        # Этап 4: Агрегация результатов
        final_report = self._compile_final_report(
            retrieval_results,
            generation_results,
            test_cases
        )
    
        return final_report

    async def evaluate_by_category(self) -> Dict[str, Any]:
        """Оценка по категориям вопросов"""
        categories = set(case.category for case in self.dataset.test_cases)
        category_results = {}

        for category in categories:
            category_cases = self.dataset.get_cases_by_category(category)
            if category_cases:
                results = await self.run_comprehensive_evaluation(category_cases)
                category_results[category] = results["aggregated_metrics"]
        
        return category_results
            
    async def evaluate_by_difficulty(self) -> Dict[str, Any]:
        """Оценка по сложности вопросов"""
        difficulties = ["easy", "medium", "hard"]
        difficulty_results = {}
        
        for difficulty in difficulties:
            difficulty_cases = self.dataset.get_cases_by_difficulty(difficulty)
            if difficulty_cases:
                results = await self.run_comprehensive_evaluation(difficulty_cases)
                difficulty_results[difficulty] = results["aggregated_metrics"]

        return difficulty_results

    def _compile_final_report(self, retrieval_results: Dict, generation_results: Dict, test_cases: List[TestCase]) -> Dict[str, Any]:
        """Компиляция финального отчета"""
        
        # Объединение метрик
        aggregated_metrics = {
            "retrieval": retrieval_results["aggregated_metrics"],
            "generation": generation_results["aggregated_metrics"],
            "overall_score": self._calculate_overall_score(
                retrieval_results["aggregated_metrics"],
                generation_results["aggregated_metrics"]
            )
        }
        
        # Детальные результаты
        detailed_results = []
        for retrieval, generation in zip(
            retrieval_results["detailed_results"],
            generation_results["detailed_results"]
        ):
            combined = {**retrieval, **generation}
            detailed_results.append(combined)
        
        report = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "test_cases_count": len(test_cases),
            "aggregated_metrics": aggregated_metrics,
            "detailed_results": detailed_results,
            "dataset_statistics": self.dataset.get_statistics()
        }
        return report
    
    def _calculate_overall_score(self, retrieval_metrics: Dict, generation_metrics: Dict) -> float:
        """Расчет общего скора системы"""
        # Веса для разных компонентов
        weights = {
            "retrieval_precision": 0.3,
            "retrieval_recall": 0.2,
            "generation_semantic_similarity": 0.3,
            "generation_rougeL": 0.2
        }

        score = 0
        score += retrieval_metrics.get("mean_precision", 0) * weights["retrieval_precision"]
        score += retrieval_metrics.get("mean_recall", 0) * weights["retrieval_recall"]
        score += generation_metrics.get("mean_semantic_similarity", 0) * weights["generation_semantic_similarity"]
        score += generation_metrics.get("mean_rougeL", 0) * weights["generation_rougeL"]
        
        return score

    def save_report(self, report: Dict[str, Any], filename: str = None):
        """Сохранение отчета в файл"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_evaluation_report_{timestamp}.json"

        # Получение пути к последней директории
        path = filename.rsplit('/', 1)[0]
        
        # Проверка директории на существование и ее создание, если она не существует
        if not os.path.isdir(path):
            os.mkdir(path)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation report saved to {filename}")
        
    def generate_summary(self, report: Dict[str, Any]) -> str:
        """Генерация текстовой сводки"""
        metrics = report["aggregated_metrics"]

        summary = f"""
📊 ОТЧЕТ ОЦЕНКИ RAG-СИСТЕМЫ
===========================

Общая информация:
- Время оценки: {report['evaluation_timestamp']}
- Количество тестовых случаев: {report['test_cases_count']}
- Общий score системы: {metrics['overall_score']:.3f}

Качество поиска (Retrieval):
- Precision: {metrics['retrieval']['mean_precision']:.3f} ± {metrics['retrieval']['std_precision']:.3f}
- Recall: {metrics['retrieval']['mean_recall']:.3f} ± {metrics['retrieval']['std_recall']:.3f}
- F1-Score: {metrics['retrieval']['mean_f1_score']:.3f} ± {metrics['retrieval']['std_f1_score']:.3f}
- MRR: {metrics['retrieval']['mean_mrr']:.3f} ± {metrics['retrieval']['std_mrr']:.3f}

Качество генерации (Generation):
- ROUGE-L: {metrics['generation']['mean_rougeL']:.3f} ± {metrics['generation']['std_rougeL']:.3f}
- BLEU: {metrics['generation']['mean_bleu']:.3f} ± {metrics['generation'] ['std_bleu']:.3f}
- Semantic Similarity: {metrics['generation']
['mean_semantic_similarity']:.3f} ± {metrics['generation']['std_semantic_similarity']:.3f}
- Средняя длина ответа: {metrics['generation']['mean_answer_length']:.1f}
слов

Рекомендации по улучшению:
{self._generate_recommendations(metrics)}
        """
        return summary
    
    def _generate_recommendations(self, metrics: Dict) -> str:
        """Генерация рекомендаций по улучшению"""
        recommendations = []
        if metrics['retrieval']['mean_precision'] < 0.7:
            recommendations.append("• Улучшить точность поиска: настроить эмбеддинг-модель или увеличить размер базы знаний")
    
        if metrics['retrieval']['mean_recall'] < 0.6:
            recommendations.append("• Увеличить полноту поиска: рассмотреть использование гибридного поиска")
    
        if metrics['generation']['mean_semantic_similarity'] < 0.7:
            recommendations.append("• Улучшить качество генерации: настроить промпты или использовать более мощную языковую модель")
    
        if metrics['generation']['mean_rougeL'] < 0.4:
            recommendations.append("• Работать над соответствием эталонным ответам: добавить few-shot примеры в промпты")
    
        if not recommendations:
            recommendations.append("• Система показывает хорошие результаты! Рекомендуется продолжить мониторинг качества.")
    
        return "\n".join(recommendations)