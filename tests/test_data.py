from kitt.data import train_test_split


def test_train_test_split():
    items = list(range(1000))
    train, test = train_test_split(items, 0.1)
    assert sorted(train + test) == items
    assert len(train) == 900
    assert len(test) == 100
