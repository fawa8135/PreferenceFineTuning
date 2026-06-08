# Libraries
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

# Load datasets
eval_loss = pd.read_csv("Curves/SFT_curves/EvalLoss.csv")
train_loss = pd.read_csv("Curves/SFT_curves/TrainLoss.csv")
eval_acc = pd.read_csv("Curves/SFT_curves/EvalAcc.csv")
train_acc = pd.read_csv("Curves/SFT_curves/TrainAcc.csv")

#sns.set_theme()

# Plots
plt.figure(figsize = (8,7))
plt.plot(train_loss["Step"], train_loss["Value"], color = "springgreen", linewidth = 1, label = "Train") #lightgreen, mediumseagreen, seagreen
plt.plot(eval_loss["Step"], eval_loss["Value"], color = "lightskyblue", linewidth = 2, label = "Validation")
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Loss", fontsize = 20)
plt.legend(fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/SFT_Loss.png")

plt.figure(figsize = (8,7))
plt.plot(train_acc["Step"], train_acc["Value"], color = "springgreen", linewidth = 1, label = "Train") #lightgreen, mediumseagreen, seagreen
plt.plot(eval_acc["Step"], eval_acc["Value"], color = "lightskyblue", linewidth = 2, label = "Validation")
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Accuracy", fontsize = 20)
plt.legend(fontsize = 16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/SFT_Accuracy.png")

