import mlflow
import mlflow.transformers
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import os

# Настройка MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Emotion-Classification-FineTuning")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    
    return {"accuracy": acc, "f1_score": f1}
    
def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=128
    )
	
def train_model(learning_rate):	
	# Загрузка и подготовка данных
	dataset = load_dataset("emotion")
	tokenized_datasets = dataset.map(tokenize_function, batched=True)
	tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
	tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
	# Параметры модели и обучения
	model_params = {
		"model_name": "distilbert-base-uncased",
		"num_labels": 6,
		"learning_rate": learning_rate,
		"batch_size": 16,
		"num_epochs": 3,
		"weight_decay": 0.01,
	}

	# Логирование параметров
	mlflow.log_params(model_params)

	# Загрузка модели
	model = AutoModelForSequenceClassification.from_pretrained(
		model_params["model_name"],
		num_labels=model_params["num_labels"],
		id2label={0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'},
		label2id={'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}
		)

	# Настройка обучения
	training_args = TrainingArguments(
		output_dir="./results",
		learning_rate=model_params["learning_rate"],
		per_device_train_batch_size=model_params["batch_size"],
		per_device_eval_batch_size=model_params["batch_size"],
		num_train_epochs=model_params["num_epochs"],
		weight_decay=model_params["weight_decay"],
		eval_strategy="epoch",
		save_strategy="epoch",
		load_best_model_at_end=True,
		metric_for_best_model="f1_score",
		logging_dir="./logs",
		logging_steps=100,
		report_to="none",
	)

	data_collator = DataCollatorWithPadding(tokenizer=AutoTokenizer.from_pretrained(model_params["model_name"]))

	# Создание тренера
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=tokenized_datasets["train"],
		eval_dataset=tokenized_datasets["validation"],
		tokenizer=AutoTokenizer.from_pretrained(model_params["model_name"]),
		data_collator=data_collator,
		compute_metrics=compute_metrics,
	)

	# Обучение с логированием метрик
	print("Начало обучения с трекингом в MLflow...")
	train_result = trainer.train()


    # Логирование метрик обучения
	evaluation = trainer.evaluate(tokenized_datasets["train"])
	results = {
		"train_loss": train_result.metrics["train_loss"],
		"eval_loss": evaluation["eval_loss"],
		"eval_accuracy": evaluation["eval_accuracy"],
		"eval_f1_score": evaluation["eval_f1_score"]
	}

	# Логирование модели в MLflow
	mlflow.transformers.log_model(
	transformers_model={
		"model": model,
		"tokenizer": AutoTokenizer.from_pretrained(model_params["model_name"])
	},
	artifact_path="emotion-classifier",
	registered_model_name="distilbert-emotion-classifier"
	)
	print("Эксперимент успешно завершен и записан в MLflow!")
	return results
