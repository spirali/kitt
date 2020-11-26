import pytest
from tensorflow import keras

from kitt.gpu.trainer import GPUTrainer


@pytest.mark.parametrize("num_gpus", [0, 1, 2])
def test_gpu_trainer_train(num_gpus):
    trainer = GPUTrainer(num_gpus=num_gpus)

    with trainer:
        model = keras.Sequential([keras.layers.Dense(4)])
        model.compile(optimizer="adam", loss="mse")

    model.fit([1, 2, 3], [3, 4, 5], batch_size=trainer.batch_size(32))


@pytest.mark.parametrize("num_gpus", [0, 1, 2])
def test_gpu_trainer_repeated_context(num_gpus):
    trainer = GPUTrainer(num_gpus=num_gpus)

    with trainer:
        pass
    with trainer:
        pass


@pytest.mark.parametrize("num_gpus", [0, 1, 2])
def test_gpu_trainer_nested_context(num_gpus):
    trainer = GPUTrainer(num_gpus=num_gpus)

    with pytest.raises(AssertionError):
        with trainer:
            with trainer:
                pass


def test_gpu_trainer_batch_size_no_gpu():
    trainer = GPUTrainer(num_gpus=0)
    assert trainer.batch_size(32) == 32
