import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .annotation import Annotation, AnnotationType, BBox


def iou(bb_a: BBox, bb_b: BBox) -> float:
    intersection = get_intersection_area(bb_a, bb_b)
    union = get_union_area(bb_a, bb_b, inter_area=intersection)
    return intersection / union


def boxes_intersect(bb_a: BBox, bb_b: BBox) -> float:
    if bb_a.xmin >= bb_b.xmax:
        return False  # A is right of B
    if bb_b.xmin >= bb_a.xmax:
        return False  # A is left of B
    if bb_a.ymax <= bb_b.ymin:
        return False  # A is above B
    if bb_a.ymin >= bb_b.ymax:
        return False  # A is below B
    return True


def get_intersection_area(bb_a: BBox, bb_b: BBox) -> float:
    if not boxes_intersect(bb_a, bb_b):
        return 0
    x_a = max(bb_a.xmin, bb_b.xmin)
    y_a = max(bb_a.ymin, bb_b.ymin)
    x_b = min(bb_a.xmax, bb_b.xmax)
    y_b = min(bb_a.ymax, bb_b.ymax)
    return (x_b - x_a) * (y_b - y_a)


def get_union_area(bb_a: BBox, bb_b: BBox, inter_area: float = None) -> float:
    area_a = bb_a.area
    area_b = bb_b.area
    if inter_area is None:
        inter_area = get_intersection_area(bb_a, bb_b)
    return float(area_a + area_b - inter_area)


def to_float_list(items) -> List[float]:
    return [float(v) for v in items]


def calculate_ap_every_point(recall: List, precision: List):
    mrec = [0.0] + list(recall) + [1.0]
    mpre = [0.0] + list(precision) + [0.0]
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0.0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return (
        float(ap),
        to_float_list(mpre[0 : len(mpre) - 1]),
        to_float_list(mrec[0 : len(mpre) - 1]),
        to_float_list(ii),
    )


@dataclass(frozen=True)
class ClassMetrics:
    precision: List[float]
    recall: List[float]
    AP: float
    interpolated_precision: List[float]
    interpolated_recall: List[float]
    total_GT: int
    total_TP: int
    total_FP: int


@dataclass(frozen=True)
class Metrics:
    per_class: Dict[str, ClassMetrics]

    @property
    def mAP(self) -> float:
        return float(np.mean([m.AP for m in self.per_class.values()]))


def get_metrics_reference(
    annotated_images: List[List[Annotation]], iou_threshold: float = 0.5
) -> Metrics:
    from src.bounding_box import BBFormat, BBType, BoundingBox
    from src.evaluators import pascal_voc_evaluator

    def to_bb(index: int, annotation: Annotation) -> BoundingBox:
        image_id = str(index)
        return BoundingBox(
            image_name=image_id,
            class_id=annotation.class_name,
            coordinates=annotation.bbox.xywh(),
            format=BBFormat.XYWH,
            bb_type=BBType.GROUND_TRUTH
            if annotation.type == AnnotationType.GROUND_TRUTH
            else BBType.DETECTED,
            confidence=annotation.confidence
            if annotation.type == AnnotationType.PREDICTION
            else None,
        )

    boxes = list(
        itertools.chain.from_iterable(
            [
                [to_bb(index, ann) for ann in annotations]
                for (index, annotations) in enumerate(annotated_images)
            ]
        )
    )

    gt_boxes = [box for box in boxes if box.get_bb_type() == BBType.GROUND_TRUTH]
    det_boxes = [box for box in boxes if box.get_bb_type() == BBType.DETECTED]
    metrics = pascal_voc_evaluator.get_pascalvoc_metrics(
        gt_boxes, det_boxes, iou_threshold=iou_threshold
    )
    return Metrics(
        per_class={
            k: ClassMetrics(
                precision=[float(v) for v in v["precision"]],
                recall=[float(v) for v in v["recall"]],
                AP=v["AP"],
                interpolated_precision=v["interpolated precision"],
                interpolated_recall=v["interpolated recall"],
                total_GT=v["total positives"],
                total_TP=int(v["total TP"]),
                total_FP=int(v["total FP"]),
            )
            for (k, v) in metrics["per_class"].items()
        }
    )


def get_metrics(
    annotated_images: List[List[Annotation]], iou_threshold: float = 0.5
) -> Metrics:
    # Structure bounding boxes per class and per annotation type
    class_to_bb = defaultdict(
        lambda: {
            AnnotationType.GROUND_TRUTH: [],
            AnnotationType.PREDICTION: [],
        }
    )
    classes_with_gt = set()
    annotation_to_image = {}

    for (image_id, annotations) in enumerate(annotated_images):
        for annotation in annotations:
            if annotation.type == AnnotationType.GROUND_TRUTH:
                classes_with_gt.add(annotation.class_name)
            class_to_bb[annotation.class_name][annotation.type].append(annotation)

            # Test that annotations are not duplicated
            assert annotation not in annotation_to_image
            annotation_to_image[annotation] = image_id

    class_metrics = {}
    # Per class precision recall calculation
    for class_id, class_annotations in class_to_bb.items():
        # Sort detections by decreasing confidence
        det_annotation_sorted = sorted(
            class_annotations[AnnotationType.PREDICTION],
            key=lambda ann: ann.confidence,
            reverse=True,
        )

        tp = np.zeros(len(det_annotation_sorted))
        fp = np.zeros(len(det_annotation_sorted))

        class_gt_annotations = class_annotations[AnnotationType.GROUND_TRUTH]
        matched_gts_per_image = defaultdict(set)

        for det_idx, det_annotation in enumerate(det_annotation_sorted):
            image_id = annotation_to_image[det_annotation]

            # Find ground truth annotations for this image
            image_gt_annotations = [
                ann
                for ann in class_gt_annotations
                if annotation_to_image[ann] == image_id
            ]
            if not image_gt_annotations:
                fp[det_idx] = 1
                continue

            # Get the maximum iou among all detections in the image
            ious = [iou(det_annotation.bbox, ann.bbox) for ann in image_gt_annotations]

            iou_max = np.max(ious)
            matched_gt = image_gt_annotations[int(np.argmax(ious))]

            # Assign detection as TP or FP
            if (
                iou_max >= iou_threshold
                and matched_gt not in matched_gts_per_image[image_id]
            ):
                tp[det_idx] = 1
                matched_gts_per_image[image_id].add(matched_gt)
            else:
                fp[det_idx] = 1

        # Compute precision, recall and average precision
        gt_count = len(class_gt_annotations)
        acc_fp = np.cumsum(fp)
        acc_tp = np.cumsum(tp)
        recall = np.ones_like(acc_fp.shape) if gt_count == 0 else acc_tp / gt_count
        precision = acc_tp / (acc_fp + acc_tp)

        ap, mpre, mrec, _ = calculate_ap_every_point(recall, precision)

        class_metrics[class_id] = ClassMetrics(
            precision=to_float_list(precision),
            recall=to_float_list(recall),
            AP=ap,
            interpolated_precision=mpre,
            interpolated_recall=mrec,
            total_GT=gt_count,
            total_TP=int(np.sum(tp)),
            total_FP=int(np.sum(fp)),
        )
    return Metrics(per_class=class_metrics)
