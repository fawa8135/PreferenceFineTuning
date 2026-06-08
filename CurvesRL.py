# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import math

# Load datasets
LossPolicyAvg = pd.read_csv("Curves/RLAI_curves/LossPolicyAvg.csv")
LossValueAvg = pd.read_csv("Curves/RLAI_curves/LossValueAvg.csv")
ObjectiveApproxKLAvg = pd.read_csv("Curves/RLAI_curves/PolicyApproxKL.csv")
ObjectiveEntropy = pd.read_csv("Curves/RLAI_curves/ObjectiveEntropy.csv")
ObjectiveKL = pd.read_csv("Curves/RLAI_curves/ObjectiveKL.csv")
ObjectiveNonScore = pd.read_csv("Curves/RLAI_curves/ObjectiveNonScore.csv")
ObjectiveRLHF = pd.read_csv("Curves/RLAI_curves/ObjectiveRLHF.csv")
ObjectiveScore = pd.read_csv("Curves/RLAI_curves/ObjectiveScore.csv")
PolicyClipfracAvg = pd.read_csv("Curves/RLAI_curves/PolicyClipfrac.csv")
ValueClipfracAvg = pd.read_csv("Curves/RLAI_curves/ValueClipfrac.csv")
PolicyEntropyAvg = pd.read_csv("Curves/RLAI_curves/PolicyEntropy.csv")
ValueRatio = pd.read_csv("Curves/RLAI_curves/Ratio.csv")
ValueRatioVar = pd.read_csv("Curves/RLAI_curves/RatioVar.csv")

#third = math.floor(LossPolicyAvg.shape[0]/3)

#print(LossPolicyAvg["Step"][0:third])

# Plots
plt.figure(figsize = (8,7))
plt.plot(ValueRatioVar["Step"], ValueRatioVar["Value"], color = "seagreen", linewidth = 2) #lightgreen, mediumseagreen, seagreen
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Ratio variance", fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/RLAI_ValueRatioVar.png")

plt.figure(figsize = (8,7))
plt.plot(ValueRatio["Step"], ValueRatio["Value"], color = "seagreen", linewidth = 2) #lightgreen, mediumseagreen, seagreen
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Ratio", fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/RLAI_ValueRatio.png")

plt.figure(figsize = (8,7))
plt.plot(PolicyEntropyAvg["Step"], PolicyEntropyAvg["Value"], color = "seagreen", linewidth = 2) #lightgreen, mediumseagreen, seagreen
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Policy entropy", fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/RLAI_PolicyEntropyAvg.png")

plt.figure(figsize = (8,7))
plt.plot(ValueClipfracAvg["Step"], ValueClipfracAvg["Value"], color = "seagreen", linewidth = 2) #lightgreen, mediumseagreen, seagreen
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Fraction clipped value updates", fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/RLAI_ValueClipfracAvg.png")

plt.figure(figsize = (8,7))
plt.plot(PolicyClipfracAvg["Step"], PolicyClipfracAvg["Value"], color = "seagreen", linewidth = 2) #lightgreen, mediumseagreen, seagreen
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Fraction clipped policy updates", fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/RLAI_PolicyClipfracAvg.png")

plt.figure(figsize = (8,7))
plt.plot(ObjectiveScore["Step"], ObjectiveScore["Value"], color = "seagreen", linewidth = 2) #lightgreen, mediumseagreen, seagreen
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Score", fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/RLAI_ObjectiveScore.png")

plt.figure(figsize = (8,7))
plt.plot(ObjectiveRLHF["Step"], ObjectiveRLHF["Value"], color = "seagreen", linewidth = 2) #lightgreen, mediumseagreen, seagreen
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Reward", fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/RLAI_ObjectiveRLHF.png")

plt.figure(figsize = (8,7))
plt.plot(ObjectiveNonScore["Step"], ObjectiveNonScore["Value"], color = "seagreen", linewidth = 2) #lightgreen, mediumseagreen, seagreen
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Non score", fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/RLAI_ObjectiveNonScore.png")

plt.figure(figsize = (8,7))
plt.plot(ObjectiveKL["Step"], ObjectiveKL["Value"], color = "seagreen", linewidth = 2) #lightgreen, mediumseagreen, seagreen
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Global KL divergence", fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/RLAI_ObjectiveKL.png")

plt.figure(figsize = (8,7))
plt.plot(ObjectiveEntropy["Step"], ObjectiveEntropy["Value"], color = "seagreen", linewidth = 2) #lightgreen, mediumseagreen, seagreen
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Entropy", fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/RLAI_ObjectiveEntropy.png")

plt.figure(figsize = (8,7))
plt.plot(ObjectiveApproxKLAvg["Step"], ObjectiveApproxKLAvg["Value"], color = "seagreen", linewidth = 2) #lightgreen, mediumseagreen, seagreen
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Local KL divergence", fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/RLAI_ObjectiveApproxKLAvg.png")

plt.figure(figsize = (8,7))
plt.plot(LossValueAvg["Step"], LossValueAvg["Value"], color = "seagreen", linewidth = 2) #lightgreen, mediumseagreen, seagreen
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Value loss", fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/RLAI_LossValueAvg.png")

plt.figure(figsize = (8,7))
plt.plot(LossPolicyAvg["Step"], LossPolicyAvg["Value"], color = "seagreen", linewidth = 2) #lightgreen, mediumseagreen, seagreen
plt.xlabel("Steps", fontsize = 20)
plt.ylabel("Policy loss", fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.tight_layout()
plt.savefig("Curves/Pictures/RLAI_LossPolicyAvg.png")