# Libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import pandas as pd
import numpy as np
from evaluate import load
import math

# Quantization
# bnb_config = BitsAndBytesConfig(
#    load_in_8bit = True,
#    bnb_8bit_quant_type = "nf8",
#    bnb_8bit_compute_dtype = torch.bfloat16
#)

# Base Model
basemodel_name = "../../models/huggingface_models/qwen/Qwen3-4B-Instruct-2507" # Qwen3-30B-A3B-Instruct-2507
basemodel = AutoModelForCausalLM.from_pretrained(
    basemodel_name, 
#   quantization_config = bnb_config,
    device_map="cuda")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(basemodel_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

# Preprocessing function
def preprocess_function(sample):
    post = sample["info"]["post"]
    idx = sample["choice"]
    summary = sample["summaries"][idx]["text"]
    message = [
        {"role": "system", "content": "You are an expert at summarizing. Please generate a shorter version of the content that contains its most important information."},
        {"role": "user", "content": post},
        {"role": "assistant", "content": summary}
    ]
    formatted_message = tokenizer.apply_chat_template(
        message,
        tokenize = False,
        add_generation_prompt = False
    )
    return {"text": formatted_message}

# AI dataset
#train = pd.read_parquet("Datasets/AI_feedback_train")  
#val = pd.read_parquet("Datasets/AI_feedback_validation")

# Train dataset
train = pd.read_parquet("Datasets/openai_summarize_from_feedback_train.parquet")
df_train = Dataset.from_pandas(train)
SFT_data = df_train.map(preprocess_function, remove_columns = ["info", "summaries", "choice", "worker", "batch", "split", "extra"])
pd_train = pd.DataFrame(SFT_data)
third_train = math.floor(pd_train.shape[0]/3)
unique_train = pd_train.iloc[0:third_train].drop_duplicates()
SFT_train = Dataset.from_pandas(unique_train)
print(SFT_train.shape)

# Validation dataset
val = pd.read_parquet("Datasets/openai_summarize_from_feedback_validation.parquet")
df_val = Dataset.from_pandas(val)
SFT_dataset = df_val.map(preprocess_function, remove_columns = ["info", "summaries", "choice", "worker", "batch", "split", "extra"])
pd_val = pd.DataFrame(SFT_dataset)
half_val = math.floor(pd_val.shape[0]/2)
unique_val = pd_val.iloc[0:half_val].drop_duplicates()
SFT_val = Dataset.from_pandas(unique_val)
print(SFT_val.shape)

# Metrics
#rouge = load("rouge")
#bert = load("bertscore")

#def compute_metrics(eval_pred):
#    prediction, prompt = eval_pred

#    predictions = np.argmax(prediction, axis = -1)
#    prompts = np.where(prompt != -100, prompt, tokenizer.pad_token_id)

#    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#    decoded_prompts = tokenizer.batch_decode(prompts, skip_special_tokens=True)
    
#    formatted_preds = [pred.strip() for pred in decoded_preds]
#    formatted_prompts = [prompt.strip() for prompt in decoded_prompts]
    
#    rouge_results = rouge.compute(
#        predictions = formatted_preds,     # Generated texts to score
#        references = formatted_prompts,    # Target texts to compare with
#        use_stemmer=True,                  # Strip word sufficies
#        use_aggregator=True                # Return mean score for all texts
#    )
    
#    results = {key: value for key, value in rouge_results.items()}

#    bert_results = bert.compute(
#        predictions = formatted_preds,            # Generated texts to score
#        references = formatted_prompts,           # Target texts to compare with
#        lang = "en",                              # Language
#        model_type = "distilbert-base-uncased",   # Model to use, test roberta-large
#        rescale_with_baseline = True              # Centering 0 as random and 1 as perfect
#    )           

#    results["bert_precision"] = np.mean(bert_results["precision"]) 
#    results["bert_recall"] = np.mean(bert_results["recall"]) 
#    results["bert_f1"] = np.mean(bert_results["f1"]) 
    
#    prediction_len = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
#    results["gen_len"] = np.mean(prediction_len)
#    return {k: round(v, 4) for k, v in results.items()}

# SFTConfig
training_args = SFTConfig(
    output_dir = "Outputs/SFT_outputs",        # Direction to save output    
    eval_strategy = "steps",                   # Steps or epoch
    eval_steps = 10,                           # Evaluate every 1000th step
    save_steps = 10,                           # Save, must be multiple of eval_steps
    do_eval = True,                            # Explicitly say to evaluate
    per_device_eval_batch_size = 16,           # Eval batch size  
    per_device_train_batch_size = 16,          # Train batch size
    num_train_epochs = 10,                     # Number of train epochs
    learning_rate = 1e-04,                     # Learning rate
    lr_scheduler_type = "cosine",              # Linear or cosine
    warmup_ratio = 0.1,                        # Ratio before peak
    lr_scheduler_kwargs = {"num_cycles": 0.5}, # Half cosine decay
    adam_beta1 = 0.9,                          # Beta 1 for first momentum estimate
    adam_beta2 = 0.999,                        # Beta 2 for second momentum estimate
    adam_epsilon = 1e-08,                      # Prevents dividion by zero
    logging_steps = 10,                        # How often to log
    report_to = "tensorboard",                 # Report to tensorboard
    max_length = 1024,                         # Max token length  
    # packing = True                           # Multiple examples are packed in the same input sequence to increase training efficiency
    load_best_model_at_end = True,             # Return the best model
    metric_for_best_model = "eval_loss",       # Choose best model based on evaluation loss
    greater_is_better = False,                 # Lower is better for loss
    save_total_limit = 2,                      # Keep only best and most recent model
    logging_first_step=True,                   # Log first step
    eval_on_start = True                       # Validate before training
)

# Lora
Lora = LoraConfig(
    r = 16,                  # Dimension of the low rank matrices
    lora_alpha = 32          # Scaling factor for LoRA layers
    # lora_dropout = 0.05    # Dropout probability for LoRA layers
    # bias = "none"          # Controls training of bias terms
)

# SFTTrainer
trainer = SFTTrainer(
    model = basemodel,                                               # Model to be trained
    args = training_args,                                            # Configuration for this trainer
    train_dataset = SFT_train,                                       # Dataset to use for training
    eval_dataset = SFT_val,                                          # Dataset to use for evaluation
#   compute_metrics = compute_metrics,                               # Function that is used to compute metrics
    peft_config = Lora,                                              # PEFT configuration
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]   # Stop training if the validation loss does not improve for 3 epochs
)

# Train the model
trainer.train()
trainer.save_model("Models/SFT_model")

# Validate the model
#results = trainer.evaluate()
#print(results)

# Loss: average cross entropy loss computed over non-masked tokens in the current logging interval
# Grad norm: L2 norm of the gradients, computed before gradient clipping
# Learning rate: current learning rate, which may change dynamically if a scheduler is used
# Entropy: average entropy of the models predicted token distribution over non-masked token
# Num tokens: total number of tokens processed so far
# Mean token accuracy: proportion of non-masked tokens for which the models top 1 prediction matches the ground truth
# Epoch: current epoch number, bsed on dataset iteration