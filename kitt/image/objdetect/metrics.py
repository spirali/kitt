from .annotation import BBox


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
