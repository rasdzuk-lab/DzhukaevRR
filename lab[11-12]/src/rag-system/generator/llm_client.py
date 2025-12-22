# generator/llm_client.py
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class LLMGenerator:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Альтернативно, можно использовать pipeline
        self.generator = pipeline(
            task="text-generation",
            model=model_name,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        logger.info(f"Loaded language model: {model_name}")
    
    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Генерация ответа на основе запроса и контекста"""

        # Формирование промта с контекстом
        context_text = self._build_context_string(context)
        prompt = self._construct_prompt(query, context_text)
        try:
        
            # Генерация ответа
            response = self.generator(
                prompt,
                max_length=512,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True
            )      
            generated_text = response[0]['generated_text']
            # Извлекаем только сгенерированную часть (после промта)
            answer = generated_text[len(prompt):].strip()
    
            return answer

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return "Извините, произошла ошибка при генерации ответа."

    def _build_context_string(self, context: List[Dict[str, Any]]) -> str:
        """Построение строки контекста из найденных документов"""
        context_parts = []
        for i, doc in enumerate(context):
            content = doc['content']
            title = doc['metadata']['title']
            score = doc['similarity_score']
            context_parts.append(f"[Документ {i+1}] {title} (схожесть {score:.3f}): {content}")
        
        return "\n\n".join(context_parts)

    def _construct_prompt(self, query: str, context: str) -> str:
        """Конструирование промта для языковой модели"""
        prompt = f"""На основе предоставленного контекста, ответь на вопрос пользователя. Если в контексте нет достаточной информации, скажи об этом. Контекст: {context} Вопрос: {query} Ответ: """
        return prompt