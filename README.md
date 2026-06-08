# Preference fine-tuning language models

## Content

`Completions` contains generated summaries

`Curves` contains the plots saved from tensorboard

`Datasets` contains the summarize from feedback datasets

`Distributions` contains plots of subreddit distributions

`Outputs` contains all data reported to tensorboard

`AIFeedback.py` code to create AI feedback dataset

`Curves**.py` code to create the plots

`DPO.py` code for DPO

`Data.py` code for data analysis

`Inference.py` code to generate and evaluate summaries

`Merge.py` code to merge LoRA adapters with base model

`RL.py` code for reinforcement learning stage

`RM.py` code for reward modeling stage

`SFT.py` code for supervised fine-tuning stage

## Workflow

1. Train SFT model from Qwen base model

2. Merge SFT adapters with base model

3. Train RM model from merged model

4. Merge RM adapters with the merged model

5. Train RL model from double merged model

6. Train DPO model from merged model

