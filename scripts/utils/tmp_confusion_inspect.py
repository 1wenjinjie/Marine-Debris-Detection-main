from pathlib import Path
from tools.plot_publication_metrics import (
    load_class_names,
    load_ground_truth,
    load_predictions,
    match_predictions,
    parse_threshold_spec,
    compute_pr_curve,
    build_confusion_matrix,
)

def main():
    pred_path = Path('runs_marida/val_yolov11n_7ch_full_return_512_resumed/predictions.json')
    labels_dir = Path('mask_bbox_rescue_split/val/labels')
    images_dir = Path('mask_bbox_rescue_split/val/images')
    data_yaml = Path('marida_mask_rescue_7ch.yaml')

    class_names = load_class_names(data_yaml)
    gts_by_image, gt_lookup, gt_counts = load_ground_truth(labels_dir, images_dir)
    preds = load_predictions(pred_path, num_classes=len(class_names), category_offset=1)
    records = match_predictions(preds, gts_by_image, 0.5)

    thresholds = parse_threshold_spec('0.05:0.95:0.01')
    pr_points = compute_pr_curve(records, thresholds, sum(gt_counts.values()))
    best_point = max(pr_points, key=lambda pt: (pt['f1'], pt['recall']))
    matrix = build_confusion_matrix(records, gt_lookup, best_point['threshold'], len(class_names))

    print('Best threshold', best_point)
    print('Class order:', class_names + ['Background'])
    for i, row in enumerate(matrix):
        label = (class_names + ['Background'])[i]
        print(label, row.tolist())

if __name__ == '__main__':
    main()
