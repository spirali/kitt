from kitt.utils.func import named


def test_named_fn():
    def foo():
        pass

    bar = named(foo, "bar")
    assert bar.__name__ == "bar"
