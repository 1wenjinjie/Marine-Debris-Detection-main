
from pathlib import Path
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import colorstr
from .mask_rescue_dataset import MaskRescueNPYDataset

class MaskRescueValidator(DetectionValidator):
    """Validator that feeds YOLO with 6/7-channel tensors."""

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Builds the dataset for validation.
        """
        dataset_root = Path(self.data.get("path", ".")).resolve()
        channels = self.data.get("channels", 3)
        
        return MaskRescueNPYDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch or self.args.batch,
            augment=False,
            hyp=self.args,
            rect=True,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=self.stride,
            pad=0.5,
            prefix=colorstr(f"{mode}: "),
            dataset_root=dataset_root,
            channels=channels,
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=1.0,
        )
