import getpass
import logging
import math
import os
import sys
import time

import requests
from tensorflow import keras


def is_announce_enabled():
    return "NO_ANNOUNCE" not in os.environ


def get_status_url():
    return os.environ.get("MM_STATUS_URL")


def send_mattermost_message(message, url):
    try:
        requests.post(url, json={"text": message})
    except BaseException as e:
        logging.warning(f"Error while announcing computation: {e}")


class AnnounceProgressCallback(keras.callbacks.Callback):
    def __init__(self, nth_epoch=1):
        """
        :param nth_epoch: Announce only every n-th epoch
        """
        super().__init__()
        self.epoch_start_time = 0
        self.nth_epoch = nth_epoch

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.nth_epoch != 0:
            return

        if not is_announce_enabled():
            return
        url = get_status_url()
        if not url:
            return

        duration = time.time() - self.epoch_start_time
        total_epochs = self.params.get("epochs", 100)
        epoch = epoch + 1
        completed_pct = epoch / total_epochs
        progress_bar = draw_progress_bar(20, completed_pct)

        args = " ".join([sys.executable] + sys.argv)
        metrics_rows = "\n".join(f"| {k} | {v} |" for (k, v) in (logs or {}).items())

        text = f"""
{getpass.getuser()}: *{args}*
Epoch {epoch} has finished in {duration:.3f} s
{epoch}/{total_epochs} ({(completed_pct * 100):.2f} %) {progress_bar}

| Metric | Value |
| :---: | :---: |
{metrics_rows}
""".strip()
        send_mattermost_message(text, url)


def draw_progress_bar(total_bars: int, percent: float) -> str:
    completed_char = "█"
    empty_char = "░"

    completed_bars = math.floor(total_bars * percent)
    return completed_char * completed_bars + empty_char * (total_bars - completed_bars)
