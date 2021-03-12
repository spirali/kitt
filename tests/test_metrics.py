from kitt.image.objdetect.annotation import BoundingBox
from kitt.image.objdetect.metrics import boxes_intersect, get_intersection_area, get_union_area
from kitt.image.objdetect.metrics import iou


def test_iou():
    bb_a = BoundingBox(0, 2, 0, 2)
    bb_b = BoundingBox(3, 5, 0, 1)
    bb_c = BoundingBox(0, 1, 3, 4)
    assert iou(bb_a, bb_b) == 0
    assert iou(bb_a, bb_c) == 0
    assert iou(bb_b, bb_c) == 0
    assert iou(bb_a, bb_a) == 1

    bb_d = BoundingBox(1, 3, 1, 3)
    assert iou(bb_a, bb_d) == get_intersection_area(bb_a, bb_d) / get_union_area(bb_a, bb_d)


def test_bb_intersect():
    bb_a = BoundingBox(0, 2, 0, 2)
    bb_b = BoundingBox(3, 5, 0, 1)
    bb_c = BoundingBox(0, 1, 3, 4)
    assert not boxes_intersect(bb_a, bb_b)
    assert not boxes_intersect(bb_a, bb_c)
    assert not boxes_intersect(bb_b, bb_c)

    bb_d = BoundingBox(2, 4, 2, 4)
    assert not boxes_intersect(bb_d, bb_a)

    bb_e = BoundingBox(1, 4, 1, 4)
    assert boxes_intersect(bb_e, bb_a)
    assert not boxes_intersect(bb_e, bb_b)
    assert not boxes_intersect(bb_e, bb_c)
    assert boxes_intersect(bb_e, bb_d)

    bb_f = BoundingBox(0, 5, 0, 5)
    assert boxes_intersect(bb_f, bb_a)
    assert boxes_intersect(bb_f, bb_b)
    assert boxes_intersect(bb_f, bb_c)
    assert boxes_intersect(bb_f, bb_d)
    assert boxes_intersect(bb_f, bb_e)


def test_get_intersection_area():
    bb_a = BoundingBox(0, 2, 0, 2)
    bb_b = BoundingBox(3, 5, 0, 1)
    bb_c = BoundingBox(0, 1, 3, 4)
    assert get_intersection_area(bb_a, bb_b) == 0
    assert get_intersection_area(bb_a, bb_c) == 0
    assert get_intersection_area(bb_b, bb_c) == 0

    bb_d = BoundingBox(1, 3, 1, 3)
    assert get_intersection_area(bb_a, bb_d) == 1
    assert get_intersection_area(bb_d, bb_a) == 1
    assert get_intersection_area(bb_a, bb_a) == 4

    bb_e = BoundingBox(0, 5, 0, 5)
    assert get_intersection_area(bb_e, bb_a) == 4
    assert get_intersection_area(bb_e, bb_b) == 2


def test_get_union_area():
    bb_a = BoundingBox(0, 2, 0, 2)
    bb_b = BoundingBox(3, 5, 0, 1)
    bb_c = BoundingBox(0, 1, 3, 4)
    assert get_union_area(bb_a, bb_b) == bb_a.area + bb_b.area
    assert get_union_area(bb_a, bb_c) == bb_a.area + bb_c.area
    assert get_union_area(bb_b, bb_c) == bb_b.area + bb_c.area
    assert get_union_area(bb_a, bb_a) == bb_a.area

    bb_d = BoundingBox(1, 3, 1, 3)
    assert get_union_area(bb_a, bb_d) == bb_a.area + bb_d.area - get_intersection_area(bb_a, bb_d)

    bb_e = BoundingBox(0, 5, 0, 5)
    assert get_union_area(bb_e, bb_b) == bb_e.area
