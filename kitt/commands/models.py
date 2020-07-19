import datetime
import os
import random
import shutil
import string
import sys
import time

import click

from ..decorators import model, command
from ..logger import logger


@model(_is_buildin=True)
@click.argument("path")
def load_model(path):
    import tensorflow as tf

    path = os.path.abspath(path)
    logger.info("Loading model '%s'", path)
    return path, tf.keras.models.load_model(path)


@command(_is_buildin=True, group="Basic")
@click.option("--epochs", default=1)
@click.option("--name")
@click.option("--overwrite", is_flag=True)
def train(state, name, epochs, overwrite):
    from tensorflow.python.keras.callbacks import ModelCheckpoint

    if not state.models:
        logger.error("No models defined")
        sys.exit(1)

    if not state.train_data:
        logger.error("No training data defined")
        sys.exit(1)

    train_data = state.train_data[-1].make_il_pair()

    if not state.test_data:
        test_data = None
    else:
        test_data = state.test_data[-1].make_il_pair()

    print(test_data[1].shape)

    model_info = state.models[-1]
    logger.debug("Model '%s' selected", model_info.name)
    model = model_info.model

    if name is None:
        now_string = datetime.datetime.now().replace(microsecond=0).isoformat()
        rnd_string = "".join(random.choice(string.ascii_lowercase) for i in range(3))
        name = "{}-{}-{}".format(model_info.name, now_string, rnd_string)

    logger.info("Starting training '%s'", name)

    train_path = os.path.abspath(name)

    if os.path.isdir(train_path):
        if overwrite:
            logger.info("Removing old training directory")
            shutil.rmtree(train_path)
        else:
            logger.critical("Training directory '%s' already exists", train_path)
            sys.exit(1)

    try:
        logger.debug("Creating path: %s", train_path)
        os.mkdir(train_path)
    except Exception as e:
        logger.critical("Cannot create training directory: %s", e)
        sys.exit(1)

    logger.info("Training %s epoch(s)", epochs)

    start_time = time.time()

    train_x, train_y = train_data

    checkpoint = ModelCheckpoint(
        os.path.join(train_path, "model.{epoch:04d}-{val_loss:.2f}"),
        save_best_only=True,
    )
    model.fit(
        train_x,
        train_y,
        epochs=epochs,
        validation_data=test_data,
        callbacks=[checkpoint],
    )

    end_time = time.time()

    logger.info("Training finished in %.1f seconds", end_time - start_time)

    model.save(os.path.join(train_path, "model"))


@command(_is_buildin=True, group="Basic")
def predict(state):
    if not state.models:
        logger.error("No models defined")
        sys.exit(1)

    if not state.train_data:
        logger.error("No training data defined")
        sys.exit(1)

    data = state.train_data[-1]
    model_info = state.models[-1]
    model = model_info.model
    logger.debug("Model '%s' selected", model_info.name)

    result = model.predict(data.inputs)
    data.predictions = result
