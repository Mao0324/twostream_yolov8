# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.models.yolo.obb.predict import OBBPredictor
from ultralytics.utils import ops


class RTDETRPredictor(BasePredictor):
    """
    RT-DETR (Real-Time Detection Transformer) Predictor extending the BasePredictor class for making predictions using
    Baidu's RT-DETR model.

    This class leverages the power of Vision Transformers to provide real-time object detection while maintaining
    high accuracy. It supports key features like efficient hybrid encoding and IoU-aware query selection.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.rtdetr import RTDETRPredictor

        args = dict(model='rtdetr-l.pt', source=ASSETS)
        predictor = RTDETRPredictor(overrides=args)
        predictor.predict_cli()
        ```

    Attributes:
        imgsz (int): Image size for inference (must be square and scale-filled).
        args (dict): Argument overrides for the predictor.
    """

    def postprocess(self, preds, img, orig_imgs):
        """
        Postprocess the raw predictions from the model to generate bounding boxes and confidence scores.

        The method filters detections based on confidence and class if specified in `self.args`.

        Args:
            preds (list): List of [predictions, extra] from the model.
            img (torch.Tensor): Processed input images.
            orig_imgs (list or torch.Tensor): Original, unprocessed images.

        Returns:
            (list[Results]): A list of Results objects containing the post-processed bounding boxes, confidence scores,
                and class labels.
        """
        if not isinstance(preds, (list, tuple)):  # list for PyTorch inference but list[0] Tensor for export inference
            preds = [preds, None]

        nd = preds[0].shape[-1]
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1, keepdim=True)  # (300, 1)
            idx = score.squeeze(-1) > self.args.conf  # (300, )
            if self.args.classes is not None:
                idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx
            pred = torch.cat([bbox, score, cls], dim=-1)[idx]  # filter
            orig_img = orig_imgs[i]
            oh, ow = orig_img.shape[:2]
            pred[..., [0, 2]] *= ow
            pred[..., [1, 3]] *= oh
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def pre_transform(self, im):
        """
        Pre-transforms the input images before feeding them into the model for inference. The input images are
        letterboxed to ensure a square aspect ratio and scale-filled. The size must be square(640) and scaleFilled.

        Args:
            im (list[np.ndarray] |torch.Tensor): Input images of shape (N,3,h,w) for tensor, [(h,w,3) x N] for list.

        Returns:
            (list): List of pre-transformed images ready for model inference.
        """
        letterbox = LetterBox(self.imgsz, auto=False, scaleFill=True)
        return [letterbox(image=x) for x in im]


class RTDETRObbPredictor(OBBPredictor):
    """RT-DETR OBB predictor using scale-filled RT-DETR preprocessing and rotated NMS."""

    @staticmethod
    def _scale_rboxes(rboxes, img_shape, ori_shape):
        """Scale xywhr boxes from the square RT-DETR input shape to native image shape."""
        if not len(rboxes):
            return rboxes
        img_h, img_w = img_shape
        ori_h, ori_w = ori_shape[:2]
        corners = ops.xywhr2xyxyxyxy(rboxes)
        corners[..., 0] *= ori_w / img_w
        corners[..., 1] *= ori_h / img_h
        return ops.xyxyxyxy2xywhr(corners.reshape(-1, 8))

    def pre_transform(self, im):
        """Pre-transform images to square scale-filled inputs required by RT-DETR decoders."""
        letterbox = LetterBox(self.imgsz, auto=False, scaleFill=True)
        return [letterbox(image=x) for x in im]

    def postprocess(self, preds, img, orig_imgs):
        """Post-process normalized RT-DETR OBB predictions into native image-space results."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
            rotated=True,
        )

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            rboxes = ops.regularize_rboxes(torch.cat([pred[:, :4], pred[:, -1:]], dim=-1))
            rboxes[:, [0, 2]] *= img.shape[3]
            rboxes[:, [1, 3]] *= img.shape[2]
            rboxes = self._scale_rboxes(rboxes, img.shape[2:], orig_img.shape)
            obb = torch.cat([rboxes, pred[:, 4:6]], dim=-1)
            results.append(Results(orig_img, path=img_path, names=self.model.names, obb=obb))
        return results
