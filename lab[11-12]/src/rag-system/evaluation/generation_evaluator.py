# evaluation/generation_evaluator.py
import evaluate
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
from .test_dataset import TestCase

class GenerationEvaluator:
    def __init__(self):
        self.rouge = evaluate.load('rouge')
        self.bleu = evaluate.load('bleu')
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

    def evaluate_generation(self, generated_answer: str, ground_truth: str) -> Dict[str, float]:
        """Оценка качества сгенерированного ответа"""
        
        # ROUGE метрики
        rouge_results = self.rouge.compute(
            predictions=[generated_answer],
            references=[ground_truth]
        )

        # BLEU метрик
        bleu_results = self.bleu.compute(
            predictions=[generated_answer],
            references=[[ground_truth]]
        )

        # Семантическая схожесть
        semantic_similarity = self._calculate_semantic_similarity(
            generated_answer, ground_truth
        )

        # Длина ответа (простейшая метрика качества)
        answer_length = len(generated_answer.split())

        return {
            "rouge1": rouge_results["rouge1"],
            "rouge2": rouge_results["rouge2"],
            "rougeL": rouge_results["rougeL"],
            "bleu": bleu_results["bleu"],
            "semantic_similarity": semantic_similarity,
            "answer_length": answer_length
        }

    def evaluate_answers(self, test_cases: List[TestCase], generated_answers: List[str]) -> Dict[str, Any]:
        """Оценка набора сгенерированных ответов"""
        results = []
        
        for test_case, generated_answer in zip(test_cases, generated_answers):
            evaluation = self.evaluate_generation(
                generated_answer,
                test_case.ground_truth_answer
            )
    
            result = {
                "question": test_case.question,
                "generated_answer": generated_answer,
                "ground_truth": test_case.ground_truth_answer,
                "category": test_case.category,
                "difficulty": test_case.difficulty
            }
            result.update(evaluation)
            results.append(result)
    
        # Агрегированные метрики
        aggregated = self._aggregate_generation_metrics(results)
        return {
            "detailed_results": results,
            "aggregated_metrics": aggregated
        }
        
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Вычисление семантической схожести через эмбеддинги"""
        embeddings = self.similarity_model.encode([text1, text2])
        similarity = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0]
        return float(similarity)
    
    def _aggregate_generation_metrics(self, results: List[Dict]) -> Dict[str, float]:
        metrics = ["rouge1", "rouge2", "rougeL", "bleu", "semantic_similarity"]
        aggregated = {}

        for metric in metrics:
            values = [r[metric] for r in results]
            aggregated[f"mean_{metric}"] = np.mean(values)
            aggregated[f"std_{metric}"] = np.std(values)
            aggregated[f"min_{metric}"] = np.min(values)
            aggregated[f"max_{metric}"] = np.max(values)

        # Дополнительные метрики
        lengths = [r["answer_length"] for r in results]
        aggregated["mean_answer_length"] = np.mean(lengths)
        
        return aggregated