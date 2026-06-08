# Libraries
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, set_seed
from datasets import Dataset
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig, PeftModel, TaskType
import pandas as pd
import math

# Set seed
seed = 42
set_seed(seed)

# Base Model
basemodel_name = "Models/Merged_model"
basemodel = AutoModelForSequenceClassification.from_pretrained(
    basemodel_name,
    num_labels = 1,
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
    preferred = sample["summaries"][idx]["text"]
    not_preferred = sample["summaries"][1-idx]["text"]

    message = [
        {"role": "system", "content": "You are an expert at summarizing. Please score the summaries based on which preserved the most important information of the content."},
        {"role": "user", "content": post}
    ]

    chosen = message + [
          {"role": "assistant", "content": preferred}]
    
    rejected = message + [
          {"role": "assistant", "content": not_preferred}]

    formatted_chosen = tokenizer.apply_chat_template(
        chosen,
        tokenize = False,
        add_generation_prompt = False
    )

    formatted_rejected = tokenizer.apply_chat_template(
        rejected,
        tokenize = False,
        add_generation_prompt = False
    )

    return {"chosen": formatted_chosen, "rejected": formatted_rejected}

# AI dataset
#train = pd.read_parquet("Datasets/AI_feedback_train")  
#val = pd.read_parquet("Datasets/AI_feedback_validation")

# Train dataset
train = pd.read_parquet("Datasets/openai_summarize_from_feedback_train.parquet")
df_train = Dataset.from_pandas(train)
RM_data = df_train.map(preprocess_function, remove_columns = ["info", "summaries", "choice", "worker", "batch", "split", "extra"])
pd_train = pd.DataFrame(RM_data)
third_train = math.floor(pd_train.shape[0]/3)
unique_train = pd_train.iloc[third_train:2*third_train]
RM_train = Dataset.from_pandas(unique_train)
print(RM_train.shape)

# Validation dataset
val = pd.read_parquet("Datasets/openai_summarize_from_feedback_validation.parquet")
df_val = Dataset.from_pandas(val)
RM_dataset = df_val.map(preprocess_function, remove_columns = ["info", "summaries", "choice", "worker", "batch", "split", "extra"])
pd_val = pd.DataFrame(RM_dataset)
half_val = math.floor(pd_val.shape[0]/2)
unique_val = pd_val.iloc[0:half_val]
RM_val = Dataset.from_pandas(unique_val)    
print(RM_val.shape)

# Reward config
training_args = RewardConfig(
    output_dir = "Outputs/Reward_outputs",           # Direction to save output 
    eval_strategy = "steps",                         # Steps or epoch
    eval_steps = 10,                                 # Evaluate every 1000th step
    save_steps = 10,                                 # Save, must be multiple of eval_steps
    do_eval = True,                                  # Explicitly say to evaluate
    per_device_eval_batch_size = 8,                  # Eval batch size  
    per_device_train_batch_size = 8,                 # Train batch size
    num_train_epochs = 10,                           # Number of train epochs
    learning_rate = 0.00002,                         # Learning rate
    lr_scheduler_type = "cosine",                    # Linear or cosine
    warmup_ratio = 0.1,                              # Ratio before peak
    lr_scheduler_kwargs = {"num_cycles": 0.5},       # Half cosine decay
    adam_beta1 = 0.95,                               # Beta 1 for first momentum estimate
    adam_beta2 = 0.999,                              # Beta 2 for second momentum estimate
    adam_epsilon = 1e-08,                            # Prevents dividion by zero
    logging_steps = 10,                              # How often to log
    report_to = "tensorboard",                       # Report to tensorboard
    max_length = 2048,                               # Max token length  
    load_best_model_at_end = True,                   # Return the best model
    metric_for_best_model = "eval_loss",             # Choose best model based on evaluation loss
    greater_is_better = False,                       # Lower is better for loss
    save_total_limit = 2,                            # Keep only best and most recent model
    logging_first_step = True,                       # Log first step
    eval_on_start = True                             # Validate before training
)

# Lora
Lora = LoraConfig(
    modules_to_save = ["score"],    # Train reward head
    r = 16,                         # Dimension of the low rank matrices
    lora_alpha = 32,                # Scaling factor for LoRA layers
    task_type = TaskType.SEQ_CLS    # Specify model
    # lora_dropout = 0.05           # Dropout probability for LoRA layers
    # bias = "none"                 # Controls training of bias terms
)

# Reward trainer
trainer = RewardTrainer(
    model = basemodel,                                               # Model to be trained
    args = training_args,                                            # Configuration for this trainer
    train_dataset = RM_train,                                        # Dataset to use for training
    eval_dataset = RM_val,                                           # Dataset to use for evaluation
    peft_config = Lora,                                              # Apply LoRA
    callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]  # Stop training if the validation loss does not improve for 3 epochs
)

# Train
trainer.train()
trainer.save_model("Models/Reward_model")

# Validate the model
#results = trainer.evaluate()
#print(results)

# Loss: average cross entropy loss over the last logging interval
# Grad norm: L2 norm of the gradients. computed before gradient clipping
# Learning rate: current learning rate, which may change dynamically if a scheduler is used
# Num tokens: total number of tokens processed so far
# Min reward: minimum reward score assigned by the model
# Mean reward: average reward score assigned by the model over the last logging interval
# Max reward: maximum reward score assigned by the model
# Accuracy: proportion of correct predictions averaged over the last logging interval
# Margin: average margin over the last logging interval
# Epoch: current epoch number, ased on dataset iteration