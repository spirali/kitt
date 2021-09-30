from kitt.data import to_onehot, train_test_split


def test_train_test_split():
    items = list(range(1000))
    train, test = train_test_split(items, 0.1)
    assert sorted(train + test) == items
    assert len(train) == 900
    assert len(test) == 100


def test_to_onehot():
    assert list(to_onehot(0, 1)) == [1]
    assert list(to_onehot(0, 2)) == [1, 0]
    assert list(to_onehot(1, 2)) == [0, 1]
    assert list(to_onehot(0, 3)) == [1, 0, 0]
    assert list(to_onehot(1, 3)) == [0, 1, 0]
    assert list(to_onehot(2, 3)) == [0, 0, 1]
