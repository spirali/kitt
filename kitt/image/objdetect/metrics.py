from typing import List

import numpy as np

from .annotation import BBox, AnnotationType, AnnotatedImage


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


def calculate_ap_every_point(recall: List, precision: List):
    mrec = [0]
    [mrec.append(e) for e in recall]
    mrec.append(1)
    mpre = [0]
    [mpre.append(e) for e in precision]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return [ap, mpre[0 : len(mpre) - 1], mrec[0 : len(mpre) - 1], ii]


def get_metrics(annotated_images: List[AnnotatedImage], iou_threshold: float = 0.5):
    ret = {}

    # Structure bounding boxes per class and per annotation type
    classes_bbs = {}  # structure {class_id: {gt: [], dt: []}}
    gt_classes_only = []  # classes with at least one GT BB
    detected_gt_per_image = {}  # {image_id: []} map
    bb_images = {}  # {BBox: image_id} map
    for annotated_image in annotated_images:
        image_id = annotated_image.filename
        n_gt = 0
        for annotation in annotated_image.annotations:
            class_id = annotation.class_name
            if class_id not in classes_bbs:
                classes_bbs[class_id] = {
                    AnnotationType.GROUND_TRUTH: [],
                    AnnotationType.PREDICTION: [],
                }
            annotation_type = annotation.annotation_type
            if (
                annotation_type == AnnotationType.GROUND_TRUTH
                and class_id not in gt_classes_only
            ):
                gt_classes_only.append(class_id)
            if annotation_type == AnnotationType.GROUND_TRUTH:
                n_gt += 1
            classes_bbs[class_id][annotation_type].append(annotation)
            bb_images[annotation.bbox] = image_id
        detected_gt_per_image[annotated_image.filename] = np.zeros(n_gt)

    # Per class precision recall calculation
    for class_id, class_annotations in classes_bbs.items():

        # Skip for BBs with no GT
        if class_id not in gt_classes_only:
            continue

        # Sort detections by decreasing confidence
        det_annotation_sorted = sorted(
            class_annotations[AnnotationType.PREDICTION],
            key=lambda annotation: annotation.confidence,
            reverse=True,
        )

        # Init TPs, FPs
        tp = np.zeros(len(det_annotation_sorted))
        fp = np.zeros(len(det_annotation_sorted))

        # Loop through detections
        for det_idx, det_annotation in enumerate(det_annotation_sorted):
            image_id = bb_images[det_annotation.bbox]

            # Find ground truth image
            gt_annotations = [
                gt_annotation
                for gt_annotation in class_annotations[AnnotationType.GROUND_TRUTH]
                if bb_images[gt_annotation.bbox] == image_id
            ]

            # Get the maximum iou among all detections in the image
            iou_max = -1.0
            best_gt_annotation_idx = None
            for gt_annotation_idx, gt_annotation in enumerate(gt_annotations):
                iou_value = iou(det_annotation.bbox, gt_annotation.bbox)
                if iou_value > iou_max:
                    iou_max = iou_value
                    best_gt_annotation_idx = gt_annotation_idx

            # Assign detection as TP or FP
            if iou_max >= iou_threshold:
                # gt was not matched with any detection
                if detected_gt_per_image[image_id][best_gt_annotation_idx] == 0:
                    tp[det_idx] = 1  # detection is set as true positive
                    detected_gt_per_image[image_id][
                        best_gt_annotation_idx
                    ] = 1  # set flag to identify gt as already 'matched'
                    # print("TP")
                else:
                    fp[det_idx] = 1  # detection is set as false positive
                    # print("FP")
            else:
                fp[det_idx] = 1  # detection is set as false positive
                # print("FP")

        # compute precision, recall and average precision
        n_pos = len(class_annotations[AnnotationType.GROUND_TRUTH])
        acc_fp = np.cumsum(fp)
        acc_tp = np.cumsum(tp)
        rec = acc_tp / n_pos
        precision = np.divide(acc_tp, (acc_fp + acc_tp))

        ap, mpre, mrec, _ = calculate_ap_every_point(rec, precision)

        # add class result in the dictionary to be returned
        ret[class_id] = {
            "precision": precision,
            "recall": rec,
            "AP": ap,
            "interpolated precision": mpre,
            "interpolated recall": mrec,
            "total positives": n_pos,
            "total TP": np.sum(tp),
            "total FP": np.sum(fp),
            "iou": iou_threshold,
        }

    # For mAP, only the classes in the gt set should be considered
    mAP = sum([v["AP"] for k, v in ret.items() if k in gt_classes_only]) / len(
        gt_classes_only
    )
    return {"per_class": ret, "mAP": mAP}
