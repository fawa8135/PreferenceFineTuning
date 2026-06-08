# Preference fine-tuning language models

## Content

`Completions` contains generated summaries

`Curves` contains the plots saved from tensorboard

`Datasets` contains the human and AI feedback datasets

`Outputs` contains all data reported to tensorboard

`AIFeedback` code to create AI feedback dataset

`Curves**` code to create the plots

`Data` code for data analysis

`DPO` code for DPO

`Inference` code to generate and evaluate summaries

`Merge` code to merge LoRA adapters with base model

`RL` code for reinforcement learning stage

`RM` code for reward modeling stage

`SFT` code for supervised fine-tuning stage

## Workflow

1. Train SFT model from Qwen base model

2. Merge SFT adapters with base model

3. Train RM model from merged model

4. Merge RM adapters with the merged model

5. Train RL model from double merged model

6. Train DPO model from merged model

