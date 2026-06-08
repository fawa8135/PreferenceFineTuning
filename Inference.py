# Libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import pandas as pd
from evaluate import load
import numpy as np
import math

# Model to evaluate
model_name = "Models/SFT_model"
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="cuda")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Preprocessing function
def preprocess_function(sample):
    post = sample["info"]["post"]
    idx = sample["choice"]
    summary = sample["summaries"][idx]["text"]

    message = [
        {"role": "system", "content": "You are an expert at summarizing. Please generate a shorter version of the content that contains its most important information."},
        {"role": "user", "content": post}
    ]

#    formatted_message = tokenizer.apply_chat_template(
#        message,
#        tokenize = False,
#        add_generation_prompt = False
#    )

    return {"text": message, "reference": summary}

# Test dataset
pd_test = pd.read_parquet("Datasets/openai_summarize_from_feedback_validation.parquet")
df_test = Dataset.from_pandas(pd_test)
dataset = df_test.map(preprocess_function, remove_columns = ["info", "summaries", "choice", "worker", "batch", "split", "extra"])
pd_dataset = pd.DataFrame(dataset)
half_test = math.floor(pd_dataset.shape[0]/2)
unique_test = pd_dataset.iloc[half_test:].drop_duplicates("text")
references = Dataset.from_pandas(unique_test)
print(references.shape)

# Generate responses
def generate(sample):
    message = sample["text"]
    
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True)
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=1024,
        do_sample=False,       
        temperature=0.0 
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return {"response": response}

# Response dataset
results = []
for i in range(len(unique_test)):
    print(i)
    sample = unique_test.iloc[i].to_dict()
    result = generate(sample)
    results.append(result)

pd_results = pd.DataFrame(results)
generated_dataset = Dataset.from_pandas(pd_results)                                                                                                         
predictions = generated_dataset["response"]
prompts = [references["reference"][i] for i in range(len(references["reference"]))]
decoded_preds = [preds.strip() for preds in predictions]
decoded_prompts = [prompt.strip() for prompt in prompts]

# Save completions
df = pd.DataFrame({
    "prompt": decoded_prompts,
    "completion": decoded_preds
})

file_name = "Completions/SFT_completions"                                                                                                                                      
df.to_parquet(file_name)

# Metrics
rouge = load("rouge")
bert = load("bertscore")
    
rouge_results = rouge.compute(
    predictions = decoded_preds,         # Generated texts to score
    references = decoded_prompts,        # Target texts to compare with
    use_stemmer=True,                    # Strip word sufficies
    use_aggregator=True)                 # Return mean score for all texts

results = {key: value for key, value in rouge_results.items()}

bert_results = bert.compute(
    predictions = decoded_preds,              # Generated texts to score
    references = decoded_prompts,             # Target texts to compare with
    lang = "en",                              # Language
    model_type = "distilbert-base-uncased",   # Model to use, test roberta-large or distilbert-base-uncased
    rescale_with_baseline = True)             # Centering 0 as random and 1 as perfect

results["bert_precision"] = np.mean(bert_results["precision"])
results["bert_recall"] = np.mean(bert_results["recall"])
results["bert_f1"] = np.mean(bert_results["f1"])
    
prediction_len = [len(tokenizer(pred).input_ids) for pred in decoded_preds]
results["gen_len"] = np.mean(prediction_len)

print({k: round(v, 4) for k, v in results.items()})




