lines = open("ultralytics_plugins/mask_rescue_dataset.py", "r", encoding="utf-8", errors="ignore").readlines()
if len(lines) > 357 and "use_classaware_cp" in lines[356]:
    lines[356] = "        use_classaware_cp = bool(cp_classes) and hyp_copy_paste > 0.0\n"
    lines[357] = "        print(f\"   use_classaware_cp: {use_classaware_cp}\")\n"
open("ultralytics_plugins/mask_rescue_dataset.py", "w", encoding="utf-8").writelines(lines)
print("Fixed indentation at lines 357-358")
