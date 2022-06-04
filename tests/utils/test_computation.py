from kitt.utils.computation import map_items


def map_fn(x):
    return x * 2


def test_map_items_serial():
    assert map_items(map_fn, [1, 2, 3], parallel=False, use_tqdm=False) == [2, 4, 6]


def test_map_items_parallel():
    assert map_items(map_fn, [1, 2, 3], parallel=True, use_tqdm=False) == [2, 4, 6]
