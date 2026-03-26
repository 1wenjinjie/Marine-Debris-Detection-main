import matplotlib.pyplot as plt
import numpy as np

# Data
classes = ['Marine Debris', 'Floating Object', 'Ship Wake', 'Overall mAP50']
baseline_scores = [0.431, 0.265, 0.339, 0.345]
p2_scores = [0.028, 0.290, 0.201, 0.173]

x = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline (6ch)', color='green')
rects2 = ax.bar(x + width/2, p2_scores, width, label='Innovation (6ch + P2)', color='orange')

ax.set_ylabel('mAP50 Score')
ax.set_title('Innovation Experiment: Impact of P2 Head')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

# Add labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('innovation_comparison.png')
print("Chart saved to innovation_comparison.png")
