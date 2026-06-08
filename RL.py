# Libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from datasets import Dataset
from trl import PPOTrainer, PPOConfig
from peft import LoraConfig, get_peft_model
import pandas as pd
import math

# SFT model
sftmodel_name = "Models/Merged_model"
sftmodel = AutoModelForCausalLM.from_pretrained(
    sftmodel_name,
    device_map="cuda")

# RL model
rlmodel_temp = AutoModelForCausalLM.from_pretrained(
    sftmodel_name,
    device_map="cuda")

Lora_policy = LoraConfig(
    r = 16,                         # Dimension of the low rank matrices
    lora_alpha = 32                 # Scaling factor
)

rlmodel = get_peft_model(rlmodel_temp, Lora_policy)

# Reward model
rewardmodel_name = "Models/DoubleMerged_model"
rewardmodel = AutoModelForSequenceClassification.from_pretrained(
    rewardmodel_name,
    device_map="cuda", 
    num_labels = 1)

# Value model
valuemodel_temp = AutoModelForSequenceClassification.from_pretrained(
    rewardmodel_name,
    device_map="cuda",
    num_labels = 1)

Lora_value = LoraConfig(
    modules_to_save = ["score"],    # Train reward head
    r = 16,                         # Dimension of the low rank matrices
    lora_alpha = 32                 # Scaling factor
)

valuemodel = get_peft_model(valuemodel_temp, Lora_value)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(sftmodel_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
rlmodel.config.pad_token_id = tokenizer.pad_token_id
rlmodel.generation_config.pad_token_id = tokenizer.pad_token_id
valuemodel.config.pad_token_id = tokenizer.pad_token_id
rewardmodel.config.pad_token_id = tokenizer.pad_token_id

# Preprocessing function
def preprocess_function(sample):
    post = sample["info"]["post"]
    idx = sample["choice"]
    preferred = sample["summaries"][idx]["text"]
    not_preferred = sample["summaries"][1-idx]["text"]

    message = [
        {"role": "system", "content": "You are an expert at summarizing. Please generate a shorter version of the content that contains its most important information."},
        {"role": "user", "content": post}
    ]
    
    formatted_message = tokenizer.apply_chat_template(
        message,
        tokenize = False,
        add_generation_prompt = False
    )

    return {"prompt": formatted_message}

# AI dataset
#train = pd.read_parquet("Datasets/AI_feedback_train")  
#val = pd.read_parquet("Datasets/AI_feedback_validation") 

# Train dataset
train = pd.read_parquet("Datasets/openai_summarize_from_feedback_train.parquet")
df_train = Dataset.from_pandas(train)
RL_data = df_train.map(preprocess_function, remove_columns = ["info", "summaries", "choice", "worker", "batch", "split", "extra"])
pd_train = pd.DataFrame(RL_data)
third_train = math.floor(pd_train.shape[0]/3)
unique_train = pd_train.iloc[2*third_train:].drop_duplicates()
RL_train = Dataset.from_pandas(unique_train)
print(RL_train.shape)

# Validation dataset
val = pd.read_parquet("Datasets/openai_summarize_from_feedback_validation.parquet")
df_val = Dataset.from_pandas(val)
RL_dataset = df_val.map(preprocess_function, remove_columns = ["info", "summaries", "choice", "worker", "batch", "split", "extra"])
pd_val = pd.DataFrame(RL_dataset)
half_val = math.floor(pd_val.shape[0]/2)
unique_val = pd_val.iloc[0:half_val].drop_duplicates()
RL_val = Dataset.from_pandas(unique_val)  
print(RL_val.shape) 

# Prepare datasets
dataset_text_field = "prompt"

def prepare_dataset(dataset, tokenizer):
    def tokenize(element):
        outputs = tokenizer(
            element[dataset_text_field],
            padding=False,
        )
        return {"input_ids": outputs["input_ids"]}

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names
    )

training = prepare_dataset(RL_train, tokenizer)
validating = prepare_dataset(RL_val, tokenizer)

# PPO config
training_args = PPOConfig(
    output_dir = "Outputs/RL_outputs",         # Direction to save output
    eval_strategy = "steps",                   # Steps or epoch
    eval_steps = 10,                           # Evaluate every 1000th step
    save_steps = 10,                           # Save, must be multiple of eval_steps
    do_eval = True,                            # Explicitly say to evaluate
    per_device_eval_batch_size = 4,            # Eval batch size  
    per_device_train_batch_size = 4,           # Train batch size
    num_train_epochs = 1,                      # Number of train epochs
    learning_rate = 5e-05,                     # Learning rate
    lr_scheduler_type = "linear",              # Linear or cosine
    warmup_ratio = 0.1,                        # Ratio before peak
#   lr_scheduler_kwargs = {"num_cycles": 0.5}, # Half cosine decay
    adam_beta1 = 0.9,                          # Beta 1 for first momentum estimate
    adam_beta2 = 0.999,                        # Beta 2 for second momentum estimate
    adam_epsilon = 1e-08,                      # Prevents dividion by zero
    logging_steps = 10,                        # How often to log
    report_to = "tensorboard",                 # Report to tensorboard
#   max_length = 1024,                         # Max token length  
#   load_best_model_at_end = True,             # Return the best model
#   metric_for_best_model = "eval_loss",       # Choose best model based on evaluation loss
#   greater_is_better = False,                 # Lower is better for loss
#   save_total_limit = 2                       # Keep only best and most recent model
    kl_coef = 0.05,                            # KL coefficient
    kl_estimator = "k3",                       # k1 - naive, k3 - potentially better
    vf_coef = 0.1,                             # Value function coefficient
    cliprange = 0.2,                           # Clip range
    gamma = 1.0,                               # Discount factor
    lam = 0.95                                 # Lambda value for GAE
)

# PPO trainer
trainer = PPOTrainer(
    model = rlmodel,                                                 # Model to be trained
    ref_model = sftmodel,                                            # Reference model used to compute the KL divergence
    value_model = valuemodel,                                        # Value model used to predict the value of a state
    reward_model = rewardmodel,                                      # Reward model used to compute the reward
    args = training_args,                                            # Configuration for this trainer
    processing_class = tokenizer,                                    # Class to process the data
    train_dataset = training,                                        # Dataset to use for training
    eval_dataset = validating                                        # Dataset to use for evaluation
#   callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]   # Stop training if the validation loss does not improve for 3 epochs
)

# Train
trainer.train()
trainer.save_model("Models/RL_model")

# Eps: tracks the number of episodes per second
# Objective/ KL: Mean KL divergence between current policy and reference policy
# Objective/ entropy: mean entropy of the policy, indicating the randomness of the actions chosen by the policy
# Objective/ non score reward: man reward from non score related sources, basically beta*KL.sum(1)
# Objective/ RLHF reward: mean RLHF reward, which is score-non score reward
# Objective/ scores: mean score returned by the reward model
# Policy/ approx KL avg_: average approximate KL divergence between consecutive PPO policies
# Policy/ clip frac avg: average fraction of policy updates that are clipped
# Loss/ policy avg: average policy loss
# Loss/ value avg: average value loss
# Value/ clip frac avg: average fraction of value function updates that are clipped
# Policy/ entropy avg: average entropy of the policy during training
# Val/ ratio: mean ratio of the current policy probability to the old policy probability
# Val/ ratio var: variance of val/ ratio
# Val/ num eos tokens: number of end of sequence tokens generated
# Lr: current learning rate
# Episode: current episode count
# Epoch: current epoch number, bsed on dataset iteration



