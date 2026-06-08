# Libraries
from datasets import Dataset
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# Preprocessing function
def preprocess_function(sample):
    post = sample["info"]["post"]
    subreddit = sample["info"]["subreddit"]
    idx = sample["choice"]
    preferred = sample["summaries"][idx]["text"]
    not_preferred = sample["summaries"][1-idx]["text"]

    message = [
        {"role": "system", "content": "You are an expert at summarizing. Please generate a shorter version of the content that contains its most important information."},
        {"role": "user", "content": post}
    ]

    chosen = message + [
          {"role": "assistant", "content": preferred}]
    
    rejected = message + [
          {"role": "assistant", "content": not_preferred}]
    
    return {"chosen": chosen, "subreddit": subreddit}                        
#    return {"chosen": chosen, "rejected": rejected, "subreddit": subreddit}        
#    return {"post": message, "subreddit": subreddit}                              
#    return {"post": post, "chosen": preferred, "rejected": not_preferred, "subreddit": subreddit}

# Load dataset
df = pd.read_parquet("Datasets/openai_summarize_from_feedback_train.parquet")                                                                                                                            
df_train = Dataset.from_pandas(df)
df_structured = df_train.map(preprocess_function, remove_columns = ["info", "summaries", "choice", "worker", "batch", "split", "extra"])
df_pd = pd.DataFrame(df_structured)

# Stage specific dataset
third = math.floor(df_pd.shape[0]/2)                                                                                                                                                                                                      
dataset = df_pd.iloc[0:third].drop_duplicates("post")
print(dataset.shape)

# Plot subreddit distribution for that dataset
counts = dataset.groupby("subreddit").count().reset_index()
sorted = counts.sort_values(by = "post", ascending = False)
subreddit = sorted["subreddit"].to_list()
count = sorted["post"].to_list()

plt.figure(figsize=(6,9))
plt.barh(subreddit, count, color = "lightgreen") #lightskyblue, lightgreen, mediumseagreen, seagreen
plt.xscale("log")
plt.xlim([1,10000])
plt.xlabel("Number of samples", fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.tight_layout()
plt.savefig("Distributions/SFT_train.png")

# Dataset properties
#print("Samples")
#print(df_pd.shape)

#print("Posts")
#print(df_pd.drop_duplicates("post").shape)

#print("Samples per post")
#print(df_pd.groupby("post").count().mean())
#print(df_pd.groupby("post").count().std())

#print("Summaries per post")
#melted = df_pd.melt(id_vars = ["post"], value_vars = ["chosen", "rejected"], value_name = "summary")
#print(melted.groupby("post")["summary"].nunique().mean())
#print(melted.groupby("post")["summary"].nunique().std())

#print("Words per post")
#df_pd["post_word_len"] = df_pd["post"].str.count(" ")+1
#print(df_pd.drop_duplicates("post")["post_word_len"].mean())
#print(df_pd.drop_duplicates("post")["post_word_len"].std())

#print("Words per chosen")
#df_pd["chosen_word_len"] = df_pd["chosen"].str.count(" ")+1
#print(df_pd.drop_duplicates("chosen")["chosen_word_len"].mean())
#print(df_pd.drop_duplicates("chosen")["chosen_word_len"].std())

#print("Words per rejected")
#df_pd["rejected_word_len"] = df_pd["rejected"].str.count(" ")+1
#print(df_pd.drop_duplicates("rejected")["rejected_word_len"].mean())
#print(df_pd.drop_duplicates("rejected")["rejected_word_len"].std())