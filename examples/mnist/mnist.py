import kitt
import click


@kitt.model()
@click.option("--size", default=128)
def simple_net(size):
    import tensorflow as tf

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(size, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile("adam", "categorical_crossentropy")
    return f"simple_net_{size}", model


@kitt.loader()
def mnist_data():
    import tensorflow as tf
    import numpy as np

    def prepare_data(data):
        x, y = data
        x = x.astype(np.float)
        y = tf.keras.utils.to_categorical(y, 10)
        return x, y

    train_data, test_data = tf.keras.datasets.mnist.load_data()
    return kitt.TrainTestPair(prepare_data(train_data), prepare_data(test_data))


@kitt.pyplot_command()
def show_data(data):
    from matplotlib import pyplot as plt

    data = data[:20]
    fig, axs = plt.subplots(len(data), 3)
    for i, (inp, lb, pr) in enumerate(data):
        if inp is not None:
            axs[i, 0].imshow(inp)
        if lb is not None:
            axs[i, 1].bar(range(10), lb)
        if pr is not None:
            print(pr)
            axs[i, 2].bar(range(10), pr)


kitt.cli_main()
