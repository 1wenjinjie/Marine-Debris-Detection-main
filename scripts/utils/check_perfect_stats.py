from pathlib import Path
from collections import Counter

def count_labels(folder):
    counts = Counter()
    path = Path(folder)
    for txt in path.glob("*.txt"):
        with open(txt, 'r') as f:
            for line in f:
                cls = int(line.split()[0])
                counts[cls] += 1
    return counts

print("Test Set Counts:")
print(count_labels("MARIDA_YOLO_Perfect/test/labels"))
print("\nTrain Set Counts:")
print(count_labels("MARIDA_YOLO_Perfect/train/labels"))
