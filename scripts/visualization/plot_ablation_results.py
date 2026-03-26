import matplotlib.pyplot as plt
import numpy as np

# Data
classes = ['Marine Debris', 'Floating Object', 'Ship Wake', 'Overall mAP50']
rgb_scores = [0.000, 0.114, 0.336, 0.150]
six_ch_scores = [0.431, 0.265, 0.339, 0.345]

x = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, rgb_scores, width, label='RGB Baseline (3ch)', color='gray')
rects2 = ax.bar(x + width/2, six_ch_scores, width, label='Multispectral (6ch)', color='green')

ax.set_ylabel('mAP50 Score')
ax.set_title('Ablation Study: RGB vs 6-Channel Multispectral')
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
plt.savefig('ablation_comparison.png')
print("Chart saved to ablation_comparison.png")
