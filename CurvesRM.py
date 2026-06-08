# Libraries
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

# Load datasets
eval_loss = pd.read_csv("Curves/RewardAI_curves/EvalLoss.csv")
train_loss = pd.read_csv("Curves/RewardAI_curves/TrainLoss.csv")
eval_acc = pd.read_csv("Curves/RewardAI_curves/EvalAcc.csv")
train_acc = pd.read_csv("Curves/RewardAI_curves/TrainAcc.csv")
eval_margin = pd.read_csv("Curves/RewardAI_curves/EvalMargin.csv")
train_margin = pd.read_csv("Curves/RewardAI_curves/TrainMargin.csv")
eval_max = pd.read_csv("Curves/RewardAI_curves/EvalMaxReward.csv")
train_max = pd.read_csv("Curves/RewardAI_curves/TrainMaxReward.csv")
eval_min = pd.read_csv("Curves/RewardAI_curves/EvalMinReward.csv")
train_min = pd.read_csv("Curves/RewardAI_curves/TrainMinReward.csv")
eval_mean = pd.read_csv("Curves/RewardAI_curves/EvalMeanReward.csv")
train_mean = pd.read_csv("Curves/RewardAI_curves/TrainMeanReward.csv")

#sns.set_theme()

# Plots
plt.figure(figsize = (8,7))
plt.plot(train_loss["Step"], train_loss["Value"], color = "mediumseagreen", linewidth = 1, label = "Train") #lightgreen, mediumseagreen, seagreen
plt.plot(eval_loss["Step"], eval_loss["Value"], color = "lightskyblue", linewidth = 2, label = "Validation")
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Loss", fontsize = 20)
plt.legend(fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/RewardAI_Loss.png")

plt.figure(figsize = (8,7))
plt.plot(train_acc["Step"], train_acc["Value"], color = "mediumseagreen", linewidth = 1, label = "Train") #lightgreen, mediumseagreen, seagreen
plt.plot(eval_acc["Step"], eval_acc["Value"], color = "lightskyblue", linewidth = 2, label = "Validation")
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Accuracy", fontsize = 20)
plt.legend(fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/RewardAI_Accuracy.png")

plt.figure(figsize = (8,7))
plt.plot(train_margin["Step"], train_margin["Value"], color = "mediumseagreen", linewidth = 1, label = "Train") #lightgreen, mediumseagreen, seagreen
plt.plot(eval_margin["Step"], eval_margin["Value"], color = "lightskyblue", linewidth = 2, label = "Validation")
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Margin", fontsize = 20)
plt.legend(fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/RewardAI_Margin.png")

plt.figure(figsize = (8,7))
plt.plot(train_max["Step"], train_max["Value"], color = "mediumseagreen", linewidth = 1, label = "Train") #lightgreen, mediumseagreen, seagreen
plt.plot(eval_max["Step"], eval_max["Value"], color = "lightskyblue", linewidth = 2, label = "Validation")
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Max reward", fontsize = 20)
plt.legend(fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/RewardAI_Max.png")

plt.figure(figsize = (8,7))
plt.plot(train_min["Step"], train_min["Value"], color = "mediumseagreen", linewidth = 1, label = "Train") #lightgreen, mediumseagreen, seagreen
plt.plot(eval_min["Step"], eval_min["Value"], color = "lightskyblue", linewidth = 2, label = "Validation")
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Min reward", fontsize = 20)
plt.legend(fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/RewardAI_Min.png")

plt.figure(figsize = (8,7))
plt.plot(train_mean["Step"], train_mean["Value"], color = "mediumseagreen", linewidth = 1, label = "Train") #lightgreen, mediumseagreen, seagreen
plt.plot(eval_mean["Step"], eval_mean["Value"], color = "lightskyblue", linewidth = 2, label = "Validation")
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Mean reward", fontsize = 20)
plt.legend(fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/RewardAI_Mean.png")






