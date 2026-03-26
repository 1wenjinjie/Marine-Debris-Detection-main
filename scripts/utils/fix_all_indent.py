lines = open("ultralytics_plugins/mask_rescue_dataset.py", "r", encoding="utf-8", errors="ignore").readlines()

# 修复lines 346-364的缩进 - 都应该是8个空格
for i in range(345, 364):  # Python 0-indexed, so 345=line 346
    if i < len(lines) and lines[i].startswith(("         ", "          ")):
        # 去掉前导空格,重新加8个
        stripped = lines[i].lstrip()
        lines[i] = "        " + stripped
        
open("ultralytics_plugins/mask_rescue_dataset.py", "w", encoding="utf-8").writelines(lines)
print("Fixed all indentation in build_transforms debug section")
