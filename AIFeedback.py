# Libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import pandas as pd
import math

# Load model
model_name = "../../models/huggingface_models/qwen/Qwen3-30B-A3B-Instruct-2507"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map = "cuda"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Genrate AI preference
def generate_preference(sample):
    post = sample["info"]["post"]
    idx = sample["choice"]
    summary_a = sample["summaries"][idx]["text"]
    summary_b = sample["summaries"][1-idx]["text"]

    prompt = f"Which summary is better? Respond with only 'A' or 'B'. Text: {post}, Summary A: {summary_a}, Summary B: {summary_b}"
    messages = [
        {"role": "user", "content": prompt}
    ]
 
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=False,       
        temperature=0.0 
    )
    
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    content = tokenizer.decode(output_ids, skip_special_tokens=True)

    return {"Response": content}

# Preference dataset
pd_train = pd.read_parquet("Datasets/openai_summarize_from_feedback_train.parquet")

results = []
for i in range(len(pd_train)):
    print(i)
    sample = pd_train.iloc[i].to_dict()
    result = generate_preference(sample)
    results.append(result)

pd_preference = pd.DataFrame(results)

# Merge datasets
pd_combined = pd.concat([pd_train, pd_preference], axis = 1)
combined = Dataset.from_pandas(pd_combined)

# Align AI feedback and dataset
def align(sample):
    AI_choice = sample["Response"]
    Human_choice = sample["choice"]
    
    if AI_choice == "A":
        sample["choice"] = Human_choice

    else:
        sample["choice"] = abs(Human_choice-1)
    
    return sample

AI_preference = combined.map(align, remove_columns = "Response")

# Save AI feedback dataset
file_name = "Datasets/AI_feedback_train"
AI_preference.to_parquet(file_name)
