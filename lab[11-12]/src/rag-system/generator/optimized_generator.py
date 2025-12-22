# generator/optimized_generator.py
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig
)
import torch
from typing import List, Dict, Any
import time
import logging

logger = logging.getLogger(__name__)

class OptimizedLLMGenerator:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.quantization_config = None
        self.generation_config = None
        self._setup_quantization()
        self._setup_generation_config()
        self._load_model()
    
    def _setup_quantization(self):
        """Настройка квантования для экономии памяти"""
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    def _setup_generation_config(self):
        """Настройка параметров генерации"""
        self.generation_config = GenerationConfig(
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=50256 # EOS token for most models
        )

    def _load_model(self):
        """Загрузка модели с оптимизациями"""
        logger.info(f"Loading optimized model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        logger.info("Model loaded successfully with optimizations")

    def generate_optimized_response(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Оптимизированная генерация ответа"""
        start_time = time.time()
        
        # Построение улучшенного промта
        prompt = self._construct_enhanced_prompt(query, context)
        
        try:
            # Токенизация с оптимизацией
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )
            
            # Генерация с кастомизированными параметрами
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config
                )
            
            # Декодирование с пропуском специальных токенов
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Извлечение ответа
            answer = generated_text[len(prompt):].strip()
            
            generation_time = time.time() - start_time
            
            return {
                "answer": answer,
                "generation_time": generation_time,
                "prompt_length": len(prompt),
                "answer_length": len(answer),
                "model": self.model_name,
                "optimized": True
            }
            
        except Exception as e:
            logger.error(f"Optimized generation failed: {e}")
            return {
                "answer": f"Generation error: {str(e)}",
                "generation_time": 0,
                "prompt_length": 0,
                "answer_length": 0,
                "model": self.model_name,
                "optimized": False
            }
    
    def _construct_enhanced_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Улучшенное конструирование промта"""
        context_text = self._build_structured_context(context)
    
        enhanced_prompt = f"""Ты - AI-ассистент, который отвечает на вопросы на основе предоставленного контекста.
        ИНСТРУКЦИИ:
        1. Используй только информацию из предоставленного контекста
        2. Если в контексте нет ответа, честно скажи об этом
        3. Будь точным и информативным
        4. Отвечай на русском языке
        КОНТЕКСТ: {context_text}
        ВОПРОС: {query}
        ОТВЕТ: """

        return enhanced_prompt
    
    def _build_structured_context(self, context: List[Dict[str, Any]]) -> str:
        """Структурированное построение контекста"""
        context_parts = []
        for i, doc in enumerate(context, 1):
            content = doc['content']
            title = doc['metadata']['title']
            category = doc['metadata']['category']
            score = doc['similarity_score']
            
            context_parts.append(
                f"Документ {i}:\n"
                f"Заголовок: {title}\n"
                f"Категория: {category}\n"
                f"Релевантность: {score:.3f}\n"
                f"Содержание: {content}\n"
            )

        return "\n" + "="*50 + "\n".join(context_parts) + "="*50