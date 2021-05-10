def create_tuple_mapper(input_fn, output_fn):
    """
    Creates a mapping function that receives a tuple (input, output) and uses the two
    provided functions to return tuple (input_fn(input), output_fn(output)).
    """

    def fun(item):
        input, output = item
        return input_fn(input), output_fn(output)

    return fun


def identity_fn(x):
    return x
