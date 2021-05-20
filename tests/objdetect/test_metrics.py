import numpy as np

from kitt.image.objdetect.annotation import Annotation, BBox
from kitt.image.objdetect.metrics import (
    boxes_intersect,
    get_intersection_area,
    get_metrics,
    get_union_area,
    iou,
)


def test_iou():
    bb_a = BBox(0, 2, 0, 2)
    bb_b = BBox(3, 5, 0, 1)
    bb_c = BBox(0, 1, 3, 4)
    assert iou(bb_a, bb_b) == 0
    assert iou(bb_a, bb_c) == 0
    assert iou(bb_b, bb_c) == 0
    assert iou(bb_a, bb_a) == 1

    bb_d = BBox(1, 3, 1, 3)
    assert iou(bb_a, bb_d) == get_intersection_area(bb_a, bb_d) / get_union_area(
        bb_a, bb_d
    )


def test_bb_intersect():
    bb_a = BBox(0, 2, 0, 2)
    bb_b = BBox(3, 5, 0, 1)
    bb_c = BBox(0, 1, 3, 4)
    assert not boxes_intersect(bb_a, bb_b)
    assert not boxes_intersect(bb_a, bb_c)
    assert not boxes_intersect(bb_b, bb_c)

    bb_d = BBox(2, 4, 2, 4)
    assert not boxes_intersect(bb_d, bb_a)

    bb_e = BBox(1, 4, 1, 4)
    assert boxes_intersect(bb_e, bb_a)
    assert not boxes_intersect(bb_e, bb_b)
    assert not boxes_intersect(bb_e, bb_c)
    assert boxes_intersect(bb_e, bb_d)

    bb_f = BBox(0, 5, 0, 5)
    assert boxes_intersect(bb_f, bb_a)
    assert boxes_intersect(bb_f, bb_b)
    assert boxes_intersect(bb_f, bb_c)
    assert boxes_intersect(bb_f, bb_d)
    assert boxes_intersect(bb_f, bb_e)


def test_get_intersection_area():
    bb_a = BBox(0, 2, 0, 2)
    bb_b = BBox(3, 5, 0, 1)
    bb_c = BBox(0, 1, 3, 4)
    assert get_intersection_area(bb_a, bb_b) == 0
    assert get_intersection_area(bb_a, bb_c) == 0
    assert get_intersection_area(bb_b, bb_c) == 0

    bb_d = BBox(1, 3, 1, 3)
    assert get_intersection_area(bb_a, bb_d) == 1
    assert get_intersection_area(bb_d, bb_a) == 1
    assert get_intersection_area(bb_a, bb_a) == 4

    bb_e = BBox(0, 5, 0, 5)
    assert get_intersection_area(bb_e, bb_a) == 4
    assert get_intersection_area(bb_e, bb_b) == 2


def test_get_union_area():
    bb_a = BBox(0, 2, 0, 2)
    bb_b = BBox(3, 5, 0, 1)
    bb_c = BBox(0, 1, 3, 4)
    assert get_union_area(bb_a, bb_b) == bb_a.area + bb_b.area
    assert get_union_area(bb_a, bb_c) == bb_a.area + bb_c.area
    assert get_union_area(bb_b, bb_c) == bb_b.area + bb_c.area
    assert get_union_area(bb_a, bb_a) == bb_a.area

    bb_d = BBox(1, 3, 1, 3)
    assert get_union_area(bb_a, bb_d) == bb_a.area + bb_d.area - get_intersection_area(
        bb_a, bb_d
    )

    bb_e = BBox(0, 5, 0, 5)
    assert get_union_area(bb_e, bb_b) == bb_e.area


def test_metrics_missing_prediction():
    bb = BBox(0, 5, 0, 5).normalize(10, 10)
    annotations = [Annotation.ground_truth("a", bb)]
    metrics = get_metrics([annotations])
    assert metrics.mAP == 0.0


def test_metrics_missing_gt():
    bb = BBox(0, 5, 0, 5).normalize(10, 10)
    annotations = [
        Annotation.prediction("a", bb, 0.9),
        Annotation.prediction("b", bb, 0.9),
        Annotation.prediction("b", bb, 0.8),
    ]
    metrics = get_metrics([annotations])
    assert metrics.per_class["a"].total_FP == 1
    assert metrics.per_class["b"].total_FP == 2
    assert metrics.mAP == 0.0


def test_metrics_perfect_prediction():
    bb = BBox(0, 5, 0, 5).normalize(10, 10)
    annotations = [
        Annotation.ground_truth("a", bb),
        Annotation.prediction("a", bb, 0.9),
    ]
    metrics = get_metrics([annotations])
    assert metrics.mAP == 1.0


def test_metrics_multiple_images_perfect_prediction():
    width, height = 10, 10
    bbox = BBox(0, 5, 0, 5).normalize(width, height)
    image_a = [
        Annotation.ground_truth("a", bbox),
        Annotation.prediction("a", bbox, 0.9),
    ]
    image_b = [
        Annotation.ground_truth("a", bbox),
        Annotation.prediction("a", bbox, 0.9),
    ]
    metrics = get_metrics([image_a, image_b])
    assert metrics.mAP == 1.0


def test_metrics_two_predictions_one_gt_1():
    width, height = 10, 10
    bbox = BBox(0, 2, 0, 2).normalize(width, height)
    annotations = [
        Annotation.ground_truth("a", bbox),
        Annotation.prediction("a", bbox, 0.5),
        Annotation.prediction("a", bbox.move(0.5, 0.5), 0.9),
    ]
    metrics = get_metrics([annotations])
    assert metrics.per_class["a"].total_FP == 1
    assert metrics.per_class["a"].total_TP == 1
    assert metrics.mAP == 0.5


def test_metrics_two_predictions_one_gt_2():
    width, height = 10, 10
    bbox = BBox(0, 2, 0, 2).normalize(width, height)
    annotations = [
        Annotation.ground_truth("a", bbox),
        Annotation.prediction("a", bbox, 0.9),
        Annotation.prediction("a", bbox.move(0.5, 0.5), 0.5),
    ]
    metrics = get_metrics([annotations])
    assert metrics.per_class["a"].total_FP == 1
    assert metrics.per_class["a"].total_TP == 1
    assert metrics.mAP == 1


def test_iou_threshold():
    bbox = BBox(0, 5, 0, 5)
    annotations = [
        Annotation.ground_truth("a", bbox),
        Annotation.prediction("a", bbox.move(2.5, 0), 0.9),
    ]
    metrics = get_metrics([annotations], iou_threshold=0.9)
    assert metrics.per_class["a"].total_FP == 1
    assert metrics.per_class["a"].total_TP == 0
    assert metrics.mAP == 0

    metrics = get_metrics([annotations], iou_threshold=0.2)
    assert metrics.per_class["a"].total_FP == 0
    assert metrics.per_class["a"].total_TP == 1
    assert metrics.mAP == 1


def test_per_class_map():
    annotations = [
        Annotation.ground_truth("a", BBox(0, 5, 0, 5)),
        Annotation.prediction("a", BBox(0, 5, 0, 5), 0.9),
        Annotation.ground_truth("b", BBox(0, 5, 0, 5)),
        Annotation.prediction("b", BBox(5, 6, 5, 6), 0.9),
    ]
    metrics = get_metrics([annotations], iou_threshold=0.9)
    assert metrics.per_class["a"].AP == 1
    assert metrics.per_class["b"].AP == 0
    assert metrics.mAP == 0.5


def test_metrics_do_not_contain_numpy_type():
    annotations = [
        Annotation.ground_truth("a", BBox(0, 5, 0, 5)),
        Annotation.prediction("a", BBox(0, 5, 0, 5), 0.9),
        Annotation.ground_truth("b", BBox(0, 5, 0, 5)),
        Annotation.prediction("b", BBox(5, 6, 5, 6), 0.9),
    ]
    metrics = get_metrics([annotations], iou_threshold=0.9)
    assert not isinstance(metrics.mAP, np.floating)
    for value in metrics.per_class.values():
        for item in value.precision:
            assert not isinstance(item, np.floating)
        for item in value.recall:
            assert not isinstance(item, np.floating)
        for item in value.interpolated_precision:
            assert not isinstance(item, np.floating)
        for item in value.interpolated_recall:
            assert not isinstance(item, np.floating)
        assert not isinstance(value.AP, np.floating)
        assert not isinstance(value.total_GT, np.integer)
        assert not isinstance(value.total_TP, np.integer)
        assert not isinstance(value.total_FP, np.integer)
