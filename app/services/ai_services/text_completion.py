import io
import os
import sys
import time

from dotenv import load_dotenv
_ = load_dotenv()
sys.path.append('./app/services/ai_services/')
sys.path[0] = './app/services/ai_services/'

from transformers import (AutoTokenizer, 
                        AutoModelForCausalLM,
                        BitsAndBytesConfig,
                        HfArgumentParser,
                        TrainingArguments,
                        logging,
                        pipeline)
import torch

from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import load_dataset

# Register
ENABLED_TASKS = os.environ.get('ENABLED_TASKS', '').split(',')

RESOURCE_CACHE = {}



bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # Activates 4-bit precision loading
    bnb_4bit_quant_type="nf4", # nf4
    bnb_4bit_compute_dtype="float16", # float16
    bnb_4bit_use_double_quant=False, # False
)

if "parrot_llm_gemma_finetuning" in ENABLED_TASKS:
    model_name = "google/gemma-7b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 quantization_config = bnb_config,
                                                 device_map = "auto")
    RESOURCE_CACHE["parrot_llm_gemma_finetuning"] = {}
    RESOURCE_CACHE["parrot_llm_gemma_finetuning"]["tokenizer"] = tokenizer
    RESOURCE_CACHE["parrot_llm_gemma_finetuning"]["model"] = model
    

if "parrot_llm_gemma_7b_task" in ENABLED_TASKS:
    print(f"[INFO] Loading Gemma 7B ...")
    model = "google/gemma-7b-it"
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline_chat = pipeline(
        "text-generation",
        model=model,
        model_kwargs={"torch_dtype": torch.float16},
        device="cuda",
    )
    RESOURCE_CACHE["parrot_llm_gemma-7b_task"] = {}
    RESOURCE_CACHE["parrot_llm_gemma-7b_task"]["tokenizer"] = tokenizer
    RESOURCE_CACHE["parrot_llm_gemma-7b_task"]["pipeline"] = pipeline_chat


def run_gemma_finetuning(data, num_train_epochs = 1,
                         per_device_train_batch_size = 1,
                         per_device_eval_batch_size = 1,
                         gradient_accumulation_steps = 1,
                         optim = "paged_adamw_32bit",
                         save_steps = 1000,
                        logging_steps = 100,
                        learning_rate = 2e-4,
                        weight_decay = 0.001,
                        fp16 = False,
                        bf16 = True,
                        max_grad_norm = 0.3,
                        max_steps = -1,
                        warmup_ratio = 0.03,
                        group_by_length = True,
                        lr_scheduler_type = "constant",
                        max_seq_length = 1024,
                        packing = True
                         ):
    
    output_dir = "parrot_llm_gemma_finetuning"
    try:
        
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj"]
        )
        
            
            # Set training parameters
        training_arguments = TrainingArguments(
            output_dir='parh output',
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            save_steps=save_steps,
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp16=fp16,
            bf16=bf16,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=group_by_length,
            lr_scheduler_type=lr_scheduler_type,
            report_to="tensorboard",
        )
        
        dataset = load_dataset("json",data_file='data.json')
        
        
        trainer = SFTTrainer(
        model=RESOURCE_CACHE["parrot_llm_gemma_finetuning"]["model"],
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        # formatting_func=format_prompts_fn,
        max_seq_length=max_seq_length,
        tokenizer=RESOURCE_CACHE["parrot_llm_gemma_finetuning"]["tokenizer"],
        args=training_arguments,
        packing=packing,
        )
        
        trainer.train()
        
        trainer.model.save_pretrained(output_dir)
    except Exception as e:
        print(e)
    return output_dir
    

def run_text_completion_gemma_7b(messages: list, configs: dict):
    if messages[0]['role'] == 'system':
        system_prompt = messages[0]['content']
        messages = messages[1:]
        prompt = RESOURCE_CACHE["parrot_llm_gemma-7b_task"]["pipeline"].tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt = f"{system_prompt}\n{prompt}"
    else:
        prompt = RESOURCE_CACHE["parrot_llm_gemma-7b_task"]["pipeline"].tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    outputs = RESOURCE_CACHE["parrot_llm_gemma-7b_task"]["pipeline"](
        prompt,
        max_new_tokens=min(configs.get("max_new_tokens", 256), 4096),
        do_sample=True,
        temperature=max(configs.get("temperature", 0.7), 0.01),
        top_k=configs.get("top_k", 50),
        top_p=configs.get("top_p", 0.95),
    )

    return outputs[0]["generated_text"][len(prompt):]
