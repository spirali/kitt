from kitt.image.objdetect.annotation import (
    BBox,
    AnnotatedImage,
    Annotation,
    AnnotationType,
)
from kitt.image.objdetect.metrics import (
    boxes_intersect,
    get_intersection_area,
    get_union_area,
    iou,
    get_metrics,
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


def test_metrics():
    width = 10
    height = 10

    bb = BBox(0, 5, 0, 5).normalize(width, height)
    image_a = AnnotatedImage(
        None,
        "a.jpg",
        [
            Annotation("a", bb, AnnotationType.GROUND_TRUTH),
            Annotation("a", bb, AnnotationType.PREDICTION, confidence=0.9),
        ],
    )
    metrics = get_metrics([image_a])
    assert metrics["mAP"] == 1.0

    bb = BBox(0, 5, 0, 5).normalize(width, height)
    image_a = AnnotatedImage(
        None, "a.jpg", [Annotation("a", bb, AnnotationType.GROUND_TRUTH)]
    )
    metrics = get_metrics([image_a])
    assert metrics["mAP"] == 0.0

    bb_a_gt = BBox(0, 5, 0, 5).normalize(width, height)
    bb_a_det = BBox(0, 5, 0, 5).normalize(width, height)
    image_a = AnnotatedImage(
        None,
        "a.jpg",
        [
            Annotation("a", bb_a_gt, AnnotationType.GROUND_TRUTH),
            Annotation("a", bb_a_det, AnnotationType.PREDICTION, confidence=0.9),
        ],
    )
    bb_b_gt = BBox(0, 5, 0, 5).normalize(width, height)
    bb_b_det = BBox(0, 5, 0, 5).normalize(width, height)
    image_b = AnnotatedImage(
        None,
        "b.jpg",
        [
            Annotation("a", bb_b_gt, AnnotationType.GROUND_TRUTH),
            Annotation("a", bb_b_det, AnnotationType.PREDICTION, confidence=0.9),
        ],
    )
    metrics = get_metrics([image_a, image_b])
    assert metrics == {}
    assert metrics["mAP"] == 1.0
