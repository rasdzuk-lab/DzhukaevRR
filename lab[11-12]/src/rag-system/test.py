from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Let's chat for 5 lines
for step in range(5):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')
    print(new_user_input_ids)
    # append the new user input tokens to the chat history
    bot_input_ids = new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))



"""pipe = pipeline("text-generation", model="microsoft/DialoGPT-medium")

pipe("This restaurant is awesome")"""

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Альтернативно, можно использовать pipeline
generator = pipeline(
        task="text-generation",
        model=model_name,
        tokenizer=model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
logger.info(f"Loaded language model: {model_name}")
        
# Генерация ответа
response = generator(
        "What do you like?",
        max_length=512,
        num_return_sequences=2,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
                
generated_text = response[0]['generated_text']
# Извлекаем только сгенерированную часть (после промта)
answer = generated_text
print(answer)

            """input_ids = self.tokenizer.encode("Как дела?" + self.tokenizer.eos_token, return_tensors='pt')
            print(input_ids)
            response = self.model.generate(
                input_ids,
                max_length=512,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=1024
            )
            generated_text = self.tokenizer.decode(response[:, input_ids.shape[-1]:][0], skip_special_tokens=True)"""