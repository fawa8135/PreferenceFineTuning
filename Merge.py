# Libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load the base model
base_model_path = "../../models/huggingface_models/qwen/Qwen3-4B-Instruct-2507"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map = "cuda"
)

# Merge base model with LoRA adapters
adapter_path = "Models/SFT_model"
model = PeftModel.from_pretrained(base_model, adapter_path)

# Save the merged model
merged_model_path = "Models/Merged_model"
merged_model = model.merge_and_unload()
merged_model.save_pretrained(merged_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(merged_model_path)