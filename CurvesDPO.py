# Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
eval_logits_chosen = pd.read_csv("Curves/DPOAI_curves/EvalLogitsChosen.csv")
train_logits_chosen = pd.read_csv("Curves/DPOAI_curves/TrainLogitsChosen.csv")
eval_logits_rejected = pd.read_csv("Curves/DPOAI_curves/EvalLogitsRejected.csv")
train_logits_rejected = pd.read_csv("Curves/DPOAI_curves/TrainLogitsRejected.csv")
eval_logps_chosen = pd.read_csv("Curves/DPOAI_curves/EvalLogpsChosen.csv")
train_logps_chosen = pd.read_csv("Curves/DPOAI_curves/TrainLogpsChosen.csv")
eval_logps_rejected = pd.read_csv("Curves/DPOAI_curves/EvalLogpsRejected.csv")
train_logps_rejected = pd.read_csv("Curves/DPOAI_curves/TrainLogpsRejected.csv")
eval_loss = pd.read_csv("Curves/DPOAI_curves/EvalLoss.csv")
train_loss = pd.read_csv("Curves/DPOAI_curves/TrainLoss.csv")
eval_rewards_acc = pd.read_csv("Curves/DPOAI_curves/EvalRewardsAcc.csv")
train_rewards_acc = pd.read_csv("Curves/DPOAI_curves/TrainRewardsAcc.csv")
eval_rewards_chosen = pd.read_csv("Curves/DPOAI_curves/EvalRewardsChosen.csv")
train_rewards_chosen = pd.read_csv("Curves/DPOAI_curves/TrainRewardsChosen.csv")
eval_rewards_margins = pd.read_csv("Curves/DPOAI_curves/EvalRewardsMargins.csv")
train_rewards_margins = pd.read_csv("Curves/DPOAI_curves/TrainRewardsMargins.csv")
eval_rewards_rejected = pd.read_csv("Curves/DPOAI_curves/EvalRewardsRejected.csv")
train_rewards_rejected = pd.read_csv("Curves/DPOAI_curves/TrainRewardsRejected.csv")

# Plots
plt.figure(figsize = (8,7))
plt.plot(train_rewards_rejected["Step"], train_rewards_rejected["Value"], color = "mediumseagreen", linewidth = 1, label = "Train") #lightgreen, mediumseagreen, seagreen
plt.plot(eval_rewards_rejected["Step"], eval_rewards_rejected["Value"], color = "lightskyblue", linewidth = 2, label = "Validation")
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Reward rejected", fontsize = 20)
plt.legend(fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/DPOAI_RewardsRejected.png")

plt.figure(figsize = (8,7))
plt.plot(train_rewards_margins["Step"], train_rewards_margins["Value"], color = "mediumseagreen", linewidth = 1, label = "Train") #lightgreen, mediumseagreen, seagreen
plt.plot(eval_rewards_margins["Step"], eval_rewards_margins["Value"], color = "lightskyblue", linewidth = 2, label = "Validation")
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Reward margin", fontsize = 20)
plt.legend(fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/DPOAI_RewardsMargin.png")

plt.figure(figsize = (8,7))
plt.plot(train_rewards_chosen["Step"], train_rewards_chosen["Value"], color = "mediumseagreen", linewidth = 1, label = "Train") #lightgreen, mediumseagreen, seagreen
plt.plot(eval_rewards_chosen["Step"], eval_rewards_chosen["Value"], color = "lightskyblue", linewidth = 2, label = "Validation")
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Reward chosen", fontsize = 20)
plt.legend(fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/DPOAI_RewardsChosen.png")

plt.figure(figsize = (8,7))
plt.plot(train_rewards_acc["Step"], train_rewards_acc["Value"], color = "mediumseagreen", linewidth = 1, label = "Train") #lightgreen, mediumseagreen, seagreen
plt.plot(eval_rewards_acc["Step"], eval_rewards_acc["Value"], color = "lightskyblue", linewidth = 2, label = "Validation")
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Accuracy", fontsize = 20)
plt.legend(fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/DPOAI_RewardsAcc.png")

plt.figure(figsize = (8,7))
plt.plot(train_loss["Step"], train_loss["Value"], color = "mediumseagreen", linewidth = 1, label = "Train") #lightgreen, mediumseagreen, seagreen
plt.plot(eval_loss["Step"], eval_loss["Value"], color = "lightskyblue", linewidth = 2, label = "Validation")
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Loss", fontsize = 20)
plt.legend(fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/DPOAI_Loss.png")

plt.figure(figsize = (8,7))
plt.plot(train_logps_rejected["Step"], train_logps_rejected["Value"], color = "mediumseagreen", linewidth = 1, label = "Train") #lightgreen, mediumseagreen, seagreen
plt.plot(eval_logps_rejected["Step"], eval_logps_rejected["Value"], color = "lightskyblue", linewidth = 2, label = "Validation")
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Log probability rejected", fontsize = 20)
plt.legend(fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/DPOAI_LogpsRejected.png")

plt.figure(figsize = (8,7))
plt.plot(train_logps_chosen["Step"], train_logps_chosen["Value"], color = "mediumseagreen", linewidth = 1, label = "Train") #lightgreen, mediumseagreen, seagreen
plt.plot(eval_logps_chosen["Step"], eval_logps_chosen["Value"], color = "lightskyblue", linewidth = 2, label = "Validation")
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Log probability chosen", fontsize = 20)
plt.legend(fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/DPOAI_LogpsChosen.png")

plt.figure(figsize = (8,7))
plt.plot(train_logits_rejected["Step"], train_logits_rejected["Value"], color = "mediumseagreen", linewidth = 1, label = "Train") #lightgreen, mediumseagreen, seagreen
plt.plot(eval_logits_rejected["Step"], eval_logits_rejected["Value"], color = "lightskyblue", linewidth = 2, label = "Validation")
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Logit rejected", fontsize = 20)
plt.legend(fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/DPOAI_LogitsRejected.png")

plt.figure(figsize = (8,7))
plt.plot(train_logits_chosen["Step"], train_logits_chosen["Value"], color = "mediumseagreen", linewidth = 1, label = "Train") #lightgreen, mediumseagreen, seagreen
plt.plot(eval_logits_chosen["Step"], eval_logits_chosen["Value"], color = "lightskyblue", linewidth = 2, label = "Validation")
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Logit chosen", fontsize = 20)
plt.legend(fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/DPOAI_LogitsChosen.png")
